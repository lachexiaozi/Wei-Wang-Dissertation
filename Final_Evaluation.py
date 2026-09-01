"""Offline Ground Truth evaluation for final fused obstacle CSV files.

The evaluator never feeds Ground Truth into detection or fusion.  It reads the
already accepted final obstacles and performs one-to-one bearing matching only
for offline TP/FP/FN measurement.

Default evaluation rule
-----------------------
* Match key: horizontal bearing in degrees.
* Gate: 1.0 degree (recorded explicitly in every output row).
* Assignment: maximum-cardinality one-to-one matching, then minimum total
  absolute bearing error among assignments with the same cardinality.
* Unmatched predictions are FP; unmatched Ground Truth objects are FN.

The 1.0-degree evaluation gate is intentionally independent from the online
fusion decision.  It is fixed here before the formal experiments and can be
overridden explicitly from the command line for a documented sensitivity run.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_CSV = ROOT / "Ground Truth Data" / "ground_truth_obstacles.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "Final Evaluation Results"
DEFAULT_EXPERIMENT_ROOTS = (
    ROOT / "Dynamic Weight Experimental Data",
    ROOT / "Fixed Weight Experimental Data",
)

GT_MATCH_TOLERANCE_DEG = 1.0
MATCHING_METHOD = (
    "Maximum-cardinality minimum-total-error one-to-one bearing assignment"
)

GROUND_TRUTH_REQUIRED_FIELDS = {
    "Layout_ID",
    "Ground_Truth_ID",
    "Object_Type",
    "Object_Name",
    "Ground_Truth_Range_m",
    "Ground_Truth_Bearing_deg",
    "Target_For_Evaluation",
}
FUSION_REQUIRED_FIELDS = {
    "Frame_ID",
    "Timestamp",
    "Scenario",
    "Weather",
    "Layout_ID",
    "Number_of_Final_Obstacles",
    "Bearing_Tolerance_deg",
    "Fusion_Threshold",
    "Fusion_Time_ms",
    "Frame_Processing_Time_ms",
}
FINAL_REQUIRED_FIELDS = {
    "Frame_ID",
    "Final_Obstacle_ID",
    "Source_Group_ID",
    "Timestamp",
    "Scenario",
    "Weather",
    "Layout_ID",
    "Representative_Bearing_deg",
    "Fusion_Score",
    "Fusion_Threshold",
    "Fusion_Decision",
    "Active_Sensors",
    "Supporting_Sensors",
}

MATCH_DETAIL_FIELDS = (
    "Method",
    "Scenario",
    "Experiment_ID",
    "Layout_ID",
    "Stage_Index",
    "Weather",
    "Frame_ID",
    "Timestamp",
    "Evaluation_Tolerance_deg",
    "Matching_Method",
    "Outcome",
    "Prediction_Obstacle_ID",
    "Prediction_Source_Group_ID",
    "Prediction_Bearing_deg",
    "Prediction_Fusion_Score",
    "Prediction_Active_Sensors",
    "Prediction_Supporting_Sensors",
    "Ground_Truth_ID",
    "Ground_Truth_Object_Type",
    "Ground_Truth_Object_Name",
    "Ground_Truth_Bearing_deg",
    "Ground_Truth_Range_m",
    "Absolute_Bearing_Error_deg",
)

FRAME_METRIC_FIELDS = (
    "Method",
    "Scenario",
    "Experiment_ID",
    "Layout_ID",
    "Stage_Index",
    "Weather",
    "Frame_ID",
    "Timestamp",
    "Evaluation_Tolerance_deg",
    "Matching_Method",
    "Ground_Truth_Count",
    "Prediction_Count",
    "TP",
    "FP",
    "FN",
    "Precision",
    "False_Detection_Rate",
    "Recall",
    "Miss_Rate",
    "F1",
    "Mean_Absolute_Bearing_Error_deg",
    "Matched_Ground_Truth_IDs",
    "Unmatched_Ground_Truth_IDs",
    "Unmatched_Prediction_IDs",
    "Prediction_Bearings_deg",
    "Ground_Truth_Bearings_deg",
    "Matched_Bearing_Pairs",
    "False_Positive_Predictions",
    "Missed_Ground_Truth",
    "Online_Bearing_Tolerance_deg",
    "Fusion_Threshold",
    "Fusion_Time_ms",
    "Frame_Processing_Time_ms",
)

SUMMARY_FIELDS = (
    "Method",
    "Scenario",
    "Experiment_ID",
    "Layout_ID",
    "Summary_Level",
    "Stage_Index",
    "Weather",
    "Evaluation_Tolerance_deg",
    "Frame_Count",
    "Start_Timestamp",
    "End_Timestamp",
    "Ground_Truth_Opportunities",
    "Prediction_Count",
    "TP",
    "FP",
    "FN",
    "Micro_Precision",
    "Micro_Recall",
    "Micro_F1",
    "Mean_Frame_Precision",
    "Mean_Frame_Recall",
    "Mean_Frame_F1",
    "Mean_Absolute_Bearing_Error_deg",
    "Mean_Predictions_Per_Frame",
    "Mean_Frame_Processing_Time_ms",
    "Estimated_Processing_FPS",
)


def _parse_bool(value: Any, field_name: str, row_number: int | None = None) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    location = "" if row_number is None else f" at row {row_number}"
    raise ValueError(f"{field_name}{location} must be True or False")


def _finite_float(value: Any, field_name: str, row_number: int | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        location = "" if row_number is None else f" at row {row_number}"
        raise ValueError(f"{field_name}{location} must be numeric") from exc
    if not math.isfinite(parsed):
        location = "" if row_number is None else f" at row {row_number}"
        raise ValueError(f"{field_name}{location} must be finite")
    return parsed


def _integer(value: Any, field_name: str, row_number: int | None = None) -> int:
    parsed = _finite_float(value, field_name, row_number)
    if not parsed.is_integer():
        location = "" if row_number is None else f" at row {row_number}"
        raise ValueError(f"{field_name}{location} must be an integer")
    return int(parsed)


def _validate_header(
    reader: csv.DictReader,
    required_fields: set[str],
    file_label: str,
) -> None:
    fields = set(reader.fieldnames or [])
    missing = required_fields - fields
    if missing:
        raise ValueError(f"{file_label} is missing fields: {sorted(missing)}")


def load_ground_truth(csv_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load every target-for-evaluation object, grouped by layout."""
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ground Truth CSV not found: {csv_path}")

    layouts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[tuple[int, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_header(reader, GROUND_TRUTH_REQUIRED_FIELDS, "Ground Truth CSV")
        for row_number, row in enumerate(reader, start=2):
            if not _parse_bool(
                row["Target_For_Evaluation"],
                "Target_For_Evaluation",
                row_number,
            ):
                continue
            layout_id = _integer(row["Layout_ID"], "Layout_ID", row_number)
            ground_truth_id = str(row["Ground_Truth_ID"]).strip()
            if not ground_truth_id:
                raise ValueError(f"Ground_Truth_ID at row {row_number} is empty")
            uniqueness_key = (layout_id, ground_truth_id)
            if uniqueness_key in seen_ids:
                raise ValueError(f"Duplicate Ground Truth ID: {uniqueness_key}")
            seen_ids.add(uniqueness_key)
            layouts[layout_id].append(
                {
                    "ground_truth_id": ground_truth_id,
                    "object_type": str(row["Object_Type"]).strip(),
                    "object_name": str(row["Object_Name"]).strip(),
                    "bearing_deg": _finite_float(
                        row["Ground_Truth_Bearing_deg"],
                        "Ground_Truth_Bearing_deg",
                        row_number,
                    ),
                    "range_m": _finite_float(
                        row["Ground_Truth_Range_m"],
                        "Ground_Truth_Range_m",
                        row_number,
                    ),
                }
            )

    if not layouts:
        raise ValueError("Ground Truth CSV contains no evaluation targets")
    for targets in layouts.values():
        targets.sort(key=lambda item: item["ground_truth_id"])
    return dict(layouts)


def _find_single_csv(experiment_directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(experiment_directory.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} under {experiment_directory}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _experiment_directory_from_final_csv(final_csv: Path) -> Path:
    parent = final_csv.parent
    if parent.name not in {
        "Final Obstacle Results Data",
        "Fixed Final Obstacle Results Data",
    }:
        raise ValueError(
            "Final obstacle CSV must be inside a recognized final-results directory: "
            f"{final_csv}"
        )
    return parent.parent


def _infer_method(final_csv: Path) -> tuple[str, str]:
    path_text = str(final_csv).lower()
    filename = final_csv.name.lower()
    if "fixed weight experimental data" in path_text or "_fixed_" in filename:
        return "Fixed Weight", "fixed_weight"
    if "dynamic weight experimental data" in path_text or "adaptive" in filename:
        return "Dynamic Weight", "dynamic_weight"
    raise ValueError(f"Unable to infer evaluation method from: {final_csv}")


def load_fusion_frames(fusion_csv: Path) -> list[dict[str, Any]]:
    """Load the complete frame index, including frames with zero predictions."""
    frames: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    with fusion_csv.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_header(reader, FUSION_REQUIRED_FIELDS, "Fusion CSV")
        previous_weather: str | None = None
        stage_index = 0
        for row_number, row in enumerate(reader, start=2):
            frame_id = _integer(row["Frame_ID"], "Frame_ID", row_number)
            if frame_id in seen_frames:
                raise ValueError(f"Duplicate fusion Frame_ID: {frame_id}")
            seen_frames.add(frame_id)
            weather = str(row["Weather"]).strip()
            if weather != previous_weather:
                stage_index += 1
                previous_weather = weather
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp": _finite_float(
                        row["Timestamp"], "Timestamp", row_number
                    ),
                    "scenario": str(row["Scenario"]).strip(),
                    "weather": weather,
                    "layout_id": _integer(
                        row["Layout_ID"], "Layout_ID", row_number
                    ),
                    "stage_index": stage_index,
                    "reported_final_count": _integer(
                        row["Number_of_Final_Obstacles"],
                        "Number_of_Final_Obstacles",
                        row_number,
                    ),
                    "online_bearing_tolerance_deg": _finite_float(
                        row["Bearing_Tolerance_deg"],
                        "Bearing_Tolerance_deg",
                        row_number,
                    ),
                    "fusion_threshold": _finite_float(
                        row["Fusion_Threshold"], "Fusion_Threshold", row_number
                    ),
                    "fusion_time_ms": _finite_float(
                        row["Fusion_Time_ms"], "Fusion_Time_ms", row_number
                    ),
                    "frame_processing_time_ms": _finite_float(
                        row["Frame_Processing_Time_ms"],
                        "Frame_Processing_Time_ms",
                        row_number,
                    ),
                }
            )
    if not frames:
        raise ValueError(f"Fusion CSV contains no frames: {fusion_csv}")
    return frames


def load_final_predictions(
    final_csv: Path,
) -> dict[int, list[dict[str, Any]]]:
    """Load accepted final obstacles, grouped by frame."""
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[tuple[int, int]] = set()
    with final_csv.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_header(reader, FINAL_REQUIRED_FIELDS, "Final obstacle CSV")
        for row_number, row in enumerate(reader, start=2):
            frame_id = _integer(row["Frame_ID"], "Frame_ID", row_number)
            obstacle_id = _integer(
                row["Final_Obstacle_ID"], "Final_Obstacle_ID", row_number
            )
            uniqueness_key = (frame_id, obstacle_id)
            if uniqueness_key in seen_ids:
                raise ValueError(f"Duplicate final obstacle ID: {uniqueness_key}")
            seen_ids.add(uniqueness_key)
            if not _parse_bool(
                row["Fusion_Decision"], "Fusion_Decision", row_number
            ):
                raise ValueError(
                    f"Rejected obstacle appears in final CSV at row {row_number}"
                )
            fusion_score = _finite_float(
                row["Fusion_Score"], "Fusion_Score", row_number
            )
            fusion_threshold = _finite_float(
                row["Fusion_Threshold"], "Fusion_Threshold", row_number
            )
            if fusion_score + 1e-12 < fusion_threshold:
                raise ValueError(
                    f"Final obstacle below fusion threshold at row {row_number}"
                )
            predictions[frame_id].append(
                {
                    "obstacle_id": obstacle_id,
                    "source_group_id": _integer(
                        row["Source_Group_ID"], "Source_Group_ID", row_number
                    ),
                    "bearing_deg": _finite_float(
                        row["Representative_Bearing_deg"],
                        "Representative_Bearing_deg",
                        row_number,
                    ),
                    "fusion_score": fusion_score,
                    "active_sensors": str(row["Active_Sensors"]).strip(),
                    "supporting_sensors": str(row["Supporting_Sensors"]).strip(),
                    "scenario": str(row["Scenario"]).strip(),
                    "weather": str(row["Weather"]).strip(),
                    "layout_id": _integer(
                        row["Layout_ID"], "Layout_ID", row_number
                    ),
                    "timestamp": _finite_float(
                        row["Timestamp"], "Timestamp", row_number
                    ),
                }
            )
    for frame_predictions in predictions.values():
        frame_predictions.sort(key=lambda item: item["obstacle_id"])
    return dict(predictions)


def _assignment_tie_key(assignment: Sequence[int | None], gt_count: int) -> tuple[int, ...]:
    return tuple(gt_count if item is None else item for item in assignment)


def optimal_one_to_one_match(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    tolerance_deg: float,
) -> list[int | None]:
    """Return the optimal Ground Truth index for each prediction, or None.

    Dynamic programming over the small Ground Truth set guarantees maximum
    TP count.  Among equal-cardinality assignments, total absolute bearing
    error is minimized.  Final ties are deterministic by Ground Truth order.
    """
    if not math.isfinite(tolerance_deg) or tolerance_deg <= 0.0:
        raise ValueError("tolerance_deg must be a finite positive number")

    # mask -> (total_error, assignment_for_processed_predictions)
    states: dict[int, tuple[float, list[int | None]]] = {0: (0.0, [])}
    gt_count = len(ground_truth)

    def keep_better(
        target: dict[int, tuple[float, list[int | None]]],
        mask: int,
        total_error: float,
        assignment: list[int | None],
    ) -> None:
        existing = target.get(mask)
        if existing is None:
            target[mask] = (total_error, assignment)
            return
        existing_error, existing_assignment = existing
        if total_error < existing_error - 1e-12:
            target[mask] = (total_error, assignment)
        elif math.isclose(total_error, existing_error, abs_tol=1e-12):
            if _assignment_tie_key(
                assignment, gt_count
            ) < _assignment_tie_key(existing_assignment, gt_count):
                target[mask] = (total_error, assignment)

    for prediction in predictions:
        next_states: dict[int, tuple[float, list[int | None]]] = {}
        prediction_bearing = float(prediction["bearing_deg"])
        for used_mask, (total_error, assignment) in states.items():
            keep_better(
                next_states,
                used_mask,
                total_error,
                [*assignment, None],
            )
            for gt_index, target in enumerate(ground_truth):
                bit = 1 << gt_index
                if used_mask & bit:
                    continue
                difference = abs(prediction_bearing - float(target["bearing_deg"]))
                if difference <= tolerance_deg + 1e-12:
                    keep_better(
                        next_states,
                        used_mask | bit,
                        total_error + difference,
                        [*assignment, gt_index],
                    )
        states = next_states

    best_mask, (_best_error, best_assignment) = min(
        states.items(),
        key=lambda item: (
            -item[0].bit_count(),
            item[1][0],
            _assignment_tie_key(item[1][1], gt_count),
        ),
    )
    if best_mask.bit_count() != sum(item is not None for item in best_assignment):
        raise AssertionError("Assignment mask disagrees with matched predictions")
    return best_assignment


def _classification_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_frame(
    method: str,
    experiment_id: str,
    frame: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    tolerance_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one frame and return detail rows plus one metric row."""
    assignment = optimal_one_to_one_match(
        predictions, ground_truth, tolerance_deg
    )
    matched_gt_indices = {index for index in assignment if index is not None}
    matched_errors: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    matched_ids: list[str] = []
    unmatched_prediction_ids: list[str] = []
    matched_bearing_pairs: list[str] = []
    false_positive_predictions: list[str] = []

    base = {
        "Method": method,
        "Scenario": frame["scenario"],
        "Experiment_ID": experiment_id,
        "Layout_ID": frame["layout_id"],
        "Stage_Index": frame["stage_index"],
        "Weather": frame["weather"],
        "Frame_ID": frame["frame_id"],
        "Timestamp": frame["timestamp"],
        "Evaluation_Tolerance_deg": tolerance_deg,
        "Matching_Method": MATCHING_METHOD,
    }

    for prediction, gt_index in zip(predictions, assignment):
        target = None if gt_index is None else ground_truth[gt_index]
        outcome = "FP" if target is None else "TP"
        difference: float | str = ""
        if target is None:
            unmatched_prediction_ids.append(str(prediction["obstacle_id"]))
            false_positive_predictions.append(
                f"P{prediction['obstacle_id']}:"
                f"{float(prediction['bearing_deg']):.6f}"
            )
        else:
            difference = abs(
                float(prediction["bearing_deg"]) - float(target["bearing_deg"])
            )
            matched_errors.append(float(difference))
            matched_ids.append(str(target["ground_truth_id"]))
            matched_bearing_pairs.append(
                f"P{prediction['obstacle_id']}:"
                f"{float(prediction['bearing_deg']):.6f}->"
                f"{target['ground_truth_id']}:"
                f"{float(target['bearing_deg']):.6f}"
                f"(error={float(difference):.6f})"
            )
        detail_rows.append(
            {
                **base,
                "Outcome": outcome,
                "Prediction_Obstacle_ID": prediction["obstacle_id"],
                "Prediction_Source_Group_ID": prediction["source_group_id"],
                "Prediction_Bearing_deg": prediction["bearing_deg"],
                "Prediction_Fusion_Score": prediction["fusion_score"],
                "Prediction_Active_Sensors": prediction["active_sensors"],
                "Prediction_Supporting_Sensors": prediction["supporting_sensors"],
                "Ground_Truth_ID": "" if target is None else target["ground_truth_id"],
                "Ground_Truth_Object_Type": "" if target is None else target["object_type"],
                "Ground_Truth_Object_Name": "" if target is None else target["object_name"],
                "Ground_Truth_Bearing_deg": "" if target is None else target["bearing_deg"],
                "Ground_Truth_Range_m": "" if target is None else target["range_m"],
                "Absolute_Bearing_Error_deg": difference,
            }
        )

    unmatched_gt_ids: list[str] = []
    missed_ground_truth: list[str] = []
    for gt_index, target in enumerate(ground_truth):
        if gt_index in matched_gt_indices:
            continue
        unmatched_gt_ids.append(str(target["ground_truth_id"]))
        missed_ground_truth.append(
            f"{target['ground_truth_id']}:"
            f"{float(target['bearing_deg']):.6f}"
        )
        detail_rows.append(
            {
                **base,
                "Outcome": "FN",
                "Prediction_Obstacle_ID": "",
                "Prediction_Source_Group_ID": "",
                "Prediction_Bearing_deg": "",
                "Prediction_Fusion_Score": "",
                "Prediction_Active_Sensors": "",
                "Prediction_Supporting_Sensors": "",
                "Ground_Truth_ID": target["ground_truth_id"],
                "Ground_Truth_Object_Type": target["object_type"],
                "Ground_Truth_Object_Name": target["object_name"],
                "Ground_Truth_Bearing_deg": target["bearing_deg"],
                "Ground_Truth_Range_m": target["range_m"],
                "Absolute_Bearing_Error_deg": "",
            }
        )

    tp = len(matched_gt_indices)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision, recall, f1 = _classification_metrics(tp, fp, fn)
    false_detection_rate = fp / (tp + fp) if tp + fp else 0.0
    miss_rate = fn / (tp + fn) if tp + fn else 0.0
    frame_metric = {
        "Method": method,
        "Scenario": frame["scenario"],
        "Experiment_ID": experiment_id,
        "Layout_ID": frame["layout_id"],
        "Stage_Index": frame["stage_index"],
        "Weather": frame["weather"],
        "Frame_ID": frame["frame_id"],
        "Timestamp": frame["timestamp"],
        "Evaluation_Tolerance_deg": tolerance_deg,
        "Matching_Method": MATCHING_METHOD,
        "Ground_Truth_Count": len(ground_truth),
        "Prediction_Count": len(predictions),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "False_Detection_Rate": false_detection_rate,
        "Recall": recall,
        "Miss_Rate": miss_rate,
        "F1": f1,
        "Mean_Absolute_Bearing_Error_deg": (
            statistics.fmean(matched_errors) if matched_errors else ""
        ),
        "Matched_Ground_Truth_IDs": "|".join(sorted(matched_ids)),
        "Unmatched_Ground_Truth_IDs": "|".join(sorted(unmatched_gt_ids)),
        "Unmatched_Prediction_IDs": "|".join(unmatched_prediction_ids),
        "Prediction_Bearings_deg": "|".join(
            f"P{prediction['obstacle_id']}:"
            f"{float(prediction['bearing_deg']):.6f}"
            for prediction in predictions
        ),
        "Ground_Truth_Bearings_deg": "|".join(
            f"{target['ground_truth_id']}:"
            f"{float(target['bearing_deg']):.6f}"
            for target in ground_truth
        ),
        "Matched_Bearing_Pairs": "|".join(matched_bearing_pairs),
        "False_Positive_Predictions": "|".join(false_positive_predictions),
        "Missed_Ground_Truth": "|".join(missed_ground_truth),
        "Online_Bearing_Tolerance_deg": frame["online_bearing_tolerance_deg"],
        "Fusion_Threshold": frame["fusion_threshold"],
        "Fusion_Time_ms": frame["fusion_time_ms"],
        "Frame_Processing_Time_ms": frame["frame_processing_time_ms"],
    }
    return detail_rows, frame_metric


def _aggregate_summary(
    frame_rows: Sequence[dict[str, Any]],
    summary_level: str,
    stage_index: int | str = "",
    weather: str = "",
) -> dict[str, Any]:
    if not frame_rows:
        raise ValueError("Cannot summarize an empty frame collection")
    first = frame_rows[0]
    tp = sum(int(row["TP"]) for row in frame_rows)
    fp = sum(int(row["FP"]) for row in frame_rows)
    fn = sum(int(row["FN"]) for row in frame_rows)
    micro_precision, micro_recall, micro_f1 = _classification_metrics(tp, fp, fn)
    total_tp = tp
    weighted_error_sum = sum(
        float(row["Mean_Absolute_Bearing_Error_deg"]) * int(row["TP"])
        for row in frame_rows
        if row["Mean_Absolute_Bearing_Error_deg"] != ""
    )
    mean_processing_ms = statistics.fmean(
        float(row["Frame_Processing_Time_ms"]) for row in frame_rows
    )
    return {
        "Method": first["Method"],
        "Scenario": first["Scenario"],
        "Experiment_ID": first["Experiment_ID"],
        "Layout_ID": first["Layout_ID"],
        "Summary_Level": summary_level,
        "Stage_Index": stage_index,
        "Weather": weather,
        "Evaluation_Tolerance_deg": first["Evaluation_Tolerance_deg"],
        "Frame_Count": len(frame_rows),
        "Start_Timestamp": min(float(row["Timestamp"]) for row in frame_rows),
        "End_Timestamp": max(float(row["Timestamp"]) for row in frame_rows),
        "Ground_Truth_Opportunities": sum(
            int(row["Ground_Truth_Count"]) for row in frame_rows
        ),
        "Prediction_Count": sum(int(row["Prediction_Count"]) for row in frame_rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1": micro_f1,
        "Mean_Frame_Precision": statistics.fmean(
            float(row["Precision"]) for row in frame_rows
        ),
        "Mean_Frame_Recall": statistics.fmean(
            float(row["Recall"]) for row in frame_rows
        ),
        "Mean_Frame_F1": statistics.fmean(float(row["F1"]) for row in frame_rows),
        "Mean_Absolute_Bearing_Error_deg": (
            weighted_error_sum / total_tp if total_tp else ""
        ),
        "Mean_Predictions_Per_Frame": statistics.fmean(
            int(row["Prediction_Count"]) for row in frame_rows
        ),
        "Mean_Frame_Processing_Time_ms": mean_processing_ms,
        "Estimated_Processing_FPS": (
            1000.0 / mean_processing_ms if mean_processing_ms > 0.0 else ""
        ),
    }


def build_summaries(
    frame_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build consecutive-stage, combined-weather, and overall summaries."""
    by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_weather: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_stage[int(row["Stage_Index"])].append(row)
        by_weather[str(row["Weather"])].append(row)

    stage_rows = [
        _aggregate_summary(
            by_stage[stage_index],
            "Stage",
            stage_index=stage_index,
            weather=str(by_stage[stage_index][0]["Weather"]),
        )
        for stage_index in sorted(by_stage)
    ]
    weather_rows = [
        _aggregate_summary(
            rows,
            "Weather",
            weather=weather,
        )
        for weather, rows in sorted(
            by_weather.items(),
            key=lambda item: min(int(row["Frame_ID"]) for row in item[1]),
        )
    ]
    overall = _aggregate_summary(frame_rows, "Overall")
    return stage_rows, weather_rows, overall


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if set(row) != set(fieldnames):
                raise AssertionError(f"Output row does not match schema for {path.name}")
            writer.writerow(row)
    temporary_path.replace(path)


def evaluate_experiment(
    final_csv: Path,
    ground_truth_by_layout: dict[int, list[dict[str, Any]]],
    output_root: Path,
    tolerance_deg: float,
) -> tuple[dict[str, Any], Path]:
    """Evaluate one experiment and write one consolidated per-frame CSV."""
    final_csv = final_csv.expanduser().resolve()
    experiment_directory = _experiment_directory_from_final_csv(final_csv)
    fusion_csv = _find_single_csv(
        experiment_directory, "*_fusion.csv", "fusion CSV"
    )
    method, method_slug = _infer_method(final_csv)
    frames = load_fusion_frames(fusion_csv)
    predictions_by_frame = load_final_predictions(final_csv)

    frame_ids = {frame["frame_id"] for frame in frames}
    unexpected_prediction_frames = set(predictions_by_frame) - frame_ids
    if unexpected_prediction_frames:
        raise ValueError(
            "Final obstacle CSV contains frames absent from Fusion CSV: "
            f"{sorted(unexpected_prediction_frames)}"
        )

    scenario_values = {frame["scenario"] for frame in frames}
    layout_values = {frame["layout_id"] for frame in frames}
    if len(scenario_values) != 1 or len(layout_values) != 1:
        raise ValueError("One experiment must contain one scenario and one layout")
    scenario = next(iter(scenario_values))
    layout_id = next(iter(layout_values))
    if layout_id not in ground_truth_by_layout:
        raise ValueError(f"No Ground Truth targets found for Layout {layout_id}")
    ground_truth = ground_truth_by_layout[layout_id]
    experiment_id = experiment_directory.name

    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        predictions = predictions_by_frame.get(frame["frame_id"], [])
        if len(predictions) != frame["reported_final_count"]:
            raise ValueError(
                f"Frame {frame['frame_id']}: final CSV count {len(predictions)} "
                f"does not equal Fusion CSV count {frame['reported_final_count']}"
            )
        for prediction in predictions:
            if (
                prediction["scenario"] != frame["scenario"]
                or prediction["weather"] != frame["weather"]
                or prediction["layout_id"] != frame["layout_id"]
                or not math.isclose(
                    prediction["timestamp"], frame["timestamp"], abs_tol=1e-9
                )
            ):
                raise ValueError(
                    f"Frame {frame['frame_id']}: final and fusion metadata disagree"
                )
        frame_details, frame_metric = evaluate_frame(
            method,
            experiment_id,
            frame,
            predictions,
            ground_truth,
            tolerance_deg,
        )
        del frame_details
        frame_rows.append(frame_metric)

    _, _, overall = build_summaries(frame_rows)
    method_directory = {
        "dynamic_weight": "Dynamic Weight Evaluation",
        "fixed_weight": "Fixed Weight Evaluation",
    }[method_slug]
    output_path = (
        output_root
        / method_directory
        / f"{scenario}_{experiment_id}_evaluation.csv"
    )
    _write_csv(output_path, FRAME_METRIC_FIELDS, frame_rows)
    return overall, output_path


def discover_final_csvs(inputs: Sequence[Path] | None) -> list[Path]:
    """Resolve explicit paths or discover every dynamic/fixed final CSV."""
    search_inputs = list(inputs or DEFAULT_EXPERIMENT_ROOTS)
    final_csvs: set[Path] = set()
    for input_path in search_inputs:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file():
            if not resolved.name.endswith("_final_obstacles.csv"):
                raise ValueError(f"Not a final-obstacle CSV: {resolved}")
            final_csvs.add(resolved)
        elif resolved.is_dir():
            final_csvs.update(resolved.rglob("*_final_obstacles.csv"))
        else:
            # Missing default roots are allowed before their first experiment;
            # an explicitly supplied missing path is not.
            if inputs is not None:
                raise FileNotFoundError(f"Experiment path not found: {resolved}")
    if not final_csvs:
        raise FileNotFoundError("No final-obstacle CSV files were discovered")
    return sorted(final_csvs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final fused obstacles against layout Ground Truth using "
            "one-to-one bearing matching."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=Path,
        help=(
            "Experiment directory, experiment root, or final-obstacle CSV. "
            "Repeat for multiple inputs. If omitted, all dynamic and fixed "
            "experiments are discovered."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help="Master Ground Truth CSV path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated evaluation CSV files.",
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=GT_MATCH_TOLERANCE_DEG,
        help="Ground Truth one-to-one bearing gate in degrees (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.tolerance_deg) or args.tolerance_deg <= 0.0:
        raise ValueError("--tolerance-deg must be a finite positive number")
    final_csvs = discover_final_csvs(args.experiment)
    ground_truth = load_ground_truth(args.ground_truth)
    output_root = args.output_root.expanduser().resolve()

    print(f"Ground Truth: {args.ground_truth.expanduser().resolve()}")
    print(f"Evaluation tolerance: {args.tolerance_deg:.6f} degrees")
    for final_csv in final_csvs:
        overall, output_path = evaluate_experiment(
            final_csv,
            ground_truth,
            output_root,
            args.tolerance_deg,
        )
        print(
            f"{overall['Method']} | {overall['Scenario']} | "
            f"{overall['Experiment_ID']} | Layout {overall['Layout_ID']} | "
            f"TP={overall['TP']} FP={overall['FP']} FN={overall['FN']} | "
            f"Precision={overall['Micro_Precision']:.6f} "
            f"Recall={overall['Micro_Recall']:.6f} "
            f"F1={overall['Micro_F1']:.6f}"
        )
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
