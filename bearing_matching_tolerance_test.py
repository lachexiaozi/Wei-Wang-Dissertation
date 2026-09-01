"""Offline bearing-tolerance sensitivity analysis.

The script applies fixed camera bearing calibrations and independently matches
CSI->LiDAR and RGB->LiDAR with greedy nearest-bearing association. It measures
coverage and candidate ambiguity only. It does not use ground truth, QCar IDs,
known obstacle positions, sensor fusion, or object grouping.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ROOT / "bearing_calibration_results_20260821_192842.csv"
DEFAULT_DETAILS_CSV = ROOT / "bearing_tolerance_matching_details.csv"
DEFAULT_SUMMARY_CSV = ROOT / "bearing_tolerance_summary.csv"

CSI_SCALE = 0.7225
CSI_OFFSET = -0.0987

RGB_SCALE = 0.9744
RGB_OFFSET = -1.1331

TOLERANCES_DEG = [0.5, 1.0, 2.0, 3.0, 5.0]
CAMERA_SENSORS = ("CSI", "RGB")

REQUIRED_INPUT_FIELDS = {
    "Frame_ID",
    "Sensor",
    "Detection_ID",
    "cluster_id",
    "bearing_deg",
}

DETAIL_FIELDS = (
    "Tolerance_deg",
    "Frame_ID",
    "Camera_Sensor",
    "Camera_Detection_ID",
    "Camera_Raw_Bearing",
    "Camera_Calibrated_Bearing",
    "Matched",
    "Matched_LiDAR_Cluster_ID",
    "LiDAR_Bearing",
    "Bearing_Difference_deg",
    "Candidate_LiDAR_Count",
)

SUMMARY_FIELDS = (
    "Tolerance_deg",
    "Sensor",
    "Total_Detections",
    "Matched",
    "Unmatched",
    "Match_Rate",
    "Mean_Difference_deg",
    "Median_Difference_deg",
    "Max_Difference_deg",
    "Ambiguous_Match_Count",
    # Mean number of in-gate LiDAR candidates per camera detection.
    "Mean_Candidate_Count",
    # Mean total camera-LiDAR in-gate candidate relationships per frame.
    "Mean_Candidate_Relationships_Per_Frame",
    # Mean unique LiDAR clusters falling within at least one gate per frame.
    "Mean_Unique_LiDAR_Candidates_Per_Frame",
)


def calibrated_camera_bearing(sensor: str, raw_bearing: float) -> float:
    """Apply the fixed linear calibration for one camera sensor."""
    if sensor == "CSI":
        return CSI_SCALE * raw_bearing + CSI_OFFSET
    if sensor == "RGB":
        return RGB_SCALE * raw_bearing + RGB_OFFSET
    raise ValueError(f"Unsupported camera sensor: {sensor}")


def _parse_identifier(value: str, field_name: str, row_number: int) -> int | str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: {field_name} is empty")
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if numeric_value.is_integer():
        return int(numeric_value)
    return text


def _parse_finite_float(value: str, field_name: str, row_number: int) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: {field_name} is empty")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: {field_name} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Row {row_number}: {field_name} is not finite")
    return parsed


def _identifier_sort_key(value: int | str) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def load_bearing_csv(
    input_path: Path,
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], list[int]]:
    """Load camera and LiDAR bearings without modifying any raw value."""
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input calibration CSV not found: {input_path}")

    frames: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"CSI": [], "RGB": [], "LiDAR": []}
    )
    seen_ids: set[tuple[int, str, int | str]] = set()

    with input_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_fields = REQUIRED_INPUT_FIELDS - fieldnames
        if missing_fields:
            raise ValueError(
                f"Input CSV is missing required fields: {sorted(missing_fields)}"
            )

        for row_number, row in enumerate(reader, start=2):
            sensor = str(row["Sensor"]).strip().upper()
            if sensor not in {"CSI", "RGB", "LIDAR"}:
                raise ValueError(f"Row {row_number}: unsupported Sensor={sensor!r}")
            sensor = "LiDAR" if sensor == "LIDAR" else sensor

            frame_id_value = _parse_identifier(
                row["Frame_ID"], "Frame_ID", row_number
            )
            if not isinstance(frame_id_value, int):
                raise ValueError(f"Row {row_number}: Frame_ID must be an integer")
            frame_id = frame_id_value
            raw_bearing = _parse_finite_float(
                row["bearing_deg"], "bearing_deg", row_number
            )

            if sensor == "LiDAR":
                identifier_text = row["cluster_id"] or row["Detection_ID"]
                detection_id = _parse_identifier(
                    identifier_text, "cluster_id", row_number
                )
                calibrated_bearing = raw_bearing
                detection = {
                    "cluster_id": detection_id,
                    "raw_bearing": raw_bearing,
                    "calibrated_bearing": calibrated_bearing,
                }
            else:
                detection_id = _parse_identifier(
                    row["Detection_ID"], "Detection_ID", row_number
                )
                calibrated_bearing = calibrated_camera_bearing(
                    sensor, raw_bearing
                )
                detection = {
                    "detection_id": detection_id,
                    "raw_bearing": raw_bearing,
                    "calibrated_bearing": calibrated_bearing,
                }

            uniqueness_key = (frame_id, sensor, detection_id)
            if uniqueness_key in seen_ids:
                raise ValueError(
                    f"Row {row_number}: duplicate frame/sensor/detection ID "
                    f"{uniqueness_key}"
                )
            seen_ids.add(uniqueness_key)
            frames[frame_id][sensor].append(detection)

    if not frames:
        raise ValueError("Input CSV contains no detections")

    frame_ids = sorted(frames)
    for frame in frames.values():
        for sensor in CAMERA_SENSORS:
            frame[sensor].sort(
                key=lambda detection: _identifier_sort_key(
                    detection["detection_id"]
                )
            )
        frame["LiDAR"].sort(
            key=lambda detection: _identifier_sort_key(detection["cluster_id"])
        )
    return dict(frames), frame_ids


def _optional_statistic(
    values: list[float], statistic: Any
) -> float | str:
    return statistic(values) if values else ""


def match_sensor_at_tolerance(
    frames: dict[int, dict[str, list[dict[str, Any]]]],
    frame_ids: Iterable[int],
    sensor: str,
    tolerance_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily match one camera sensor to LiDAR independently by frame."""
    detail_rows: list[dict[str, Any]] = []
    matched_differences: list[float] = []
    candidate_counts: list[int] = []
    ambiguous_count = 0
    candidate_relationships_per_frame: list[int] = []
    unique_lidar_candidates_per_frame: list[int] = []

    for frame_id in frame_ids:
        camera_detections = frames[frame_id][sensor]
        lidar_detections = frames[frame_id]["LiDAR"]
        available_lidar = {
            detection["cluster_id"]: detection for detection in lidar_detections
        }
        frame_candidate_relationships = 0
        frame_unique_candidate_ids: set[int | str] = set()

        for camera in camera_detections:
            camera_bearing = camera["calibrated_bearing"]

            # Ambiguity is a property of the tolerance gate itself, so count
            # all in-gate clusters before greedy matching consumes any cluster.
            in_gate_candidates = [
                lidar
                for lidar in lidar_detections
                if abs(camera_bearing - lidar["calibrated_bearing"])
                <= tolerance_deg
            ]
            candidate_count = len(in_gate_candidates)
            candidate_counts.append(candidate_count)
            frame_candidate_relationships += candidate_count
            frame_unique_candidate_ids.update(
                lidar["cluster_id"] for lidar in in_gate_candidates
            )
            if candidate_count >= 2:
                ambiguous_count += 1

            nearest = None
            if available_lidar:
                nearest = min(
                    available_lidar.values(),
                    key=lambda lidar: (
                        abs(camera_bearing - lidar["calibrated_bearing"]),
                        _identifier_sort_key(lidar["cluster_id"]),
                    ),
                )

            matched = False
            matched_cluster_id: int | str = ""
            lidar_bearing: float | str = ""
            bearing_difference: float | str = ""
            if nearest is not None:
                difference = abs(
                    camera_bearing - nearest["calibrated_bearing"]
                )
                if difference <= tolerance_deg:
                    matched = True
                    matched_cluster_id = nearest["cluster_id"]
                    lidar_bearing = nearest["calibrated_bearing"]
                    bearing_difference = difference
                    matched_differences.append(difference)
                    del available_lidar[nearest["cluster_id"]]

            detail_rows.append(
                {
                    "Tolerance_deg": tolerance_deg,
                    "Frame_ID": frame_id,
                    "Camera_Sensor": sensor,
                    "Camera_Detection_ID": camera["detection_id"],
                    "Camera_Raw_Bearing": camera["raw_bearing"],
                    "Camera_Calibrated_Bearing": camera_bearing,
                    "Matched": matched,
                    "Matched_LiDAR_Cluster_ID": matched_cluster_id,
                    "LiDAR_Bearing": lidar_bearing,
                    "Bearing_Difference_deg": bearing_difference,
                    "Candidate_LiDAR_Count": candidate_count,
                }
            )

        candidate_relationships_per_frame.append(frame_candidate_relationships)
        unique_lidar_candidates_per_frame.append(len(frame_unique_candidate_ids))

    total_detections = len(detail_rows)
    matched_count = len(matched_differences)
    unmatched_count = total_detections - matched_count
    summary = {
        "Tolerance_deg": tolerance_deg,
        "Sensor": sensor,
        "Total_Detections": total_detections,
        "Matched": matched_count,
        "Unmatched": unmatched_count,
        "Match_Rate": (
            matched_count / total_detections if total_detections else 0.0
        ),
        "Mean_Difference_deg": _optional_statistic(
            matched_differences, statistics.mean
        ),
        "Median_Difference_deg": _optional_statistic(
            matched_differences, statistics.median
        ),
        "Max_Difference_deg": (
            max(matched_differences) if matched_differences else ""
        ),
        "Ambiguous_Match_Count": ambiguous_count,
        "Mean_Candidate_Count": (
            statistics.mean(candidate_counts) if candidate_counts else 0.0
        ),
        "Mean_Candidate_Relationships_Per_Frame": statistics.mean(
            candidate_relationships_per_frame
        ),
        "Mean_Unique_LiDAR_Candidates_Per_Frame": statistics.mean(
            unique_lidar_candidates_per_frame
        ),
    }
    return detail_rows, summary


def analyze_all_tolerances(
    frames: dict[int, dict[str, list[dict[str, Any]]]],
    frame_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_details = []
    all_summaries = []
    for tolerance_deg in TOLERANCES_DEG:
        for sensor in CAMERA_SENSORS:
            details, summary = match_sensor_at_tolerance(
                frames, frame_ids, sensor, tolerance_deg
            )
            all_details.extend(details)
            all_summaries.append(summary)
    return all_details, all_summaries


def write_csv(
    output_path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary_table(summaries: Iterable[dict[str, Any]]) -> None:
    print(
        "Tolerance | Sensor | Match Rate | Mean delta | Max delta | Ambiguous"
    )
    print("-" * 74)
    for summary in summaries:
        mean_difference = summary["Mean_Difference_deg"]
        max_difference = summary["Max_Difference_deg"]
        mean_text = (
            f"{mean_difference:.3f} deg"
            if isinstance(mean_difference, (int, float))
            else "N/A"
        )
        max_text = (
            f"{max_difference:.3f} deg"
            if isinstance(max_difference, (int, float))
            else "N/A"
        )
        print(
            f"{summary['Tolerance_deg']:>8.1f} deg | "
            f"{summary['Sensor']:<6} | "
            f"{summary['Match_Rate']:>9.2%} | "
            f"{mean_text:>10} | "
            f"{max_text:>9} | "
            f"{summary['Ambiguous_Match_Count']:>9}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline greedy camera-LiDAR bearing matching tolerance analysis."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"input calibration CSV (default: {DEFAULT_INPUT_CSV.name})",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_DETAILS_CSV,
        help=f"details CSV (default: {DEFAULT_DETAILS_CSV.name})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"summary CSV (default: {DEFAULT_SUMMARY_CSV.name})",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        input_path = args.input.expanduser().resolve()
        details_path = args.details_output.expanduser().resolve()
        summary_path = args.summary_output.expanduser().resolve()
        if input_path in {details_path, summary_path}:
            raise ValueError("An output path cannot overwrite the input CSV")
        if details_path == summary_path:
            raise ValueError("Details and summary output paths must be different")

        frames, frame_ids = load_bearing_csv(input_path)
        details, summaries = analyze_all_tolerances(frames, frame_ids)
        write_csv(details_path, DETAIL_FIELDS, details)
        write_csv(summary_path, SUMMARY_FIELDS, summaries)
        print_summary_table(summaries)
        print(f"\nDetails CSV: {details_path}")
        print(f"Summary CSV: {summary_path}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
