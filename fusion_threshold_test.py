"""Compare three decision thresholds on existing real adaptive fusion scores.

Ground truth is intentionally absent: it is neither needed nor used for
association, fusion, or this acceptance-count comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from adaptive_object_fusion import adaptive_object_fusion


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ROOT / "Dynamic Weight Experimental Data"
DEFAULT_OUTPUT_PATH = ROOT / "fusion_threshold_comparison.csv"

FUSION_THRESHOLDS = [0.4, 0.5, 0.6]
CSV_FIELDS = (
    "Threshold",
    "Total_Groups",
    "Accepted_Groups",
    "Rejected_Groups",
    "Acceptance_Rate",
)


def _parse_bool(value: Any, field_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field_name} must be True or False, got {value!r}")


def _controller_result_from_row(row: dict[str, str]) -> dict[str, Any]:
    controller_result: dict[str, Any] = {}
    for sensor_key, csv_prefix in (
        ("csi", "CSI"),
        ("rgb", "RGB"),
        ("lidar", "LiDAR"),
    ):
        controller_result[sensor_key] = {
            "quality_score": float(row[f"{csv_prefix}_Quality"]),
            "state": row[f"{csv_prefix}_State"],
            "active": _parse_bool(
                row[f"{csv_prefix}_Active"], f"{csv_prefix}_Active"
            ),
            "weight": float(row[f"{csv_prefix}_Weight"]),
        }
    controller_result["no_active_sensor"] = not any(
        controller_result[sensor]["active"]
        for sensor in ("csi", "rgb", "lidar")
    )
    return controller_result


def _detection_outputs_from_row(
    row: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    outputs = {
        "csi": json.loads(row["CSI_Detections_JSON"]),
        "rgb": json.loads(row["RGB_Detections_JSON"]),
        "lidar": json.loads(row["LiDAR_Detections_JSON"]),
    }
    for sensor, detections in outputs.items():
        if not isinstance(detections, list):
            raise TypeError(f"{sensor} detections must decode to a list")
    return outputs


def load_existing_fusion_scores(input_root: Path) -> list[float]:
    """Run existing fusion once per real frame with no decision threshold."""
    input_paths = sorted(input_root.rglob("*_result.csv"))
    if not input_paths:
        raise FileNotFoundError(
            f"No real detection-result CSV files found below {input_root}"
        )

    fusion_scores: list[float] = []
    for input_path in input_paths:
        with input_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                result = adaptive_object_fusion(
                    frame_id=int(row["Frame_ID"]),
                    controller_result=_controller_result_from_row(row),
                    detection_outputs=_detection_outputs_from_row(row),
                    fusion_threshold=None,
                )
                if result["Fusion_Threshold"] is not None:
                    raise AssertionError("Fusion must run without a decision threshold")
                for group in result["Object_Groups"]:
                    if group["Fusion_Decision"] is not None:
                        raise AssertionError(
                            "Threshold decision was applied during fusion"
                        )
                    fusion_scores.append(float(group["Fusion_Score"]))
    return fusion_scores


def compare_thresholds(fusion_scores: Iterable[float]) -> list[dict[str, Any]]:
    """Apply only Fusion_Score >= threshold to already computed scores."""
    scores = list(fusion_scores)
    total_groups = len(scores)
    rows: list[dict[str, Any]] = []
    for threshold in FUSION_THRESHOLDS:
        fusion_decisions = [score >= threshold for score in scores]
        accepted_groups = sum(fusion_decisions)
        rejected_groups = total_groups - accepted_groups
        acceptance_rate = accepted_groups / total_groups if total_groups else 0.0
        rows.append(
            {
                "Threshold": threshold,
                "Total_Groups": total_groups,
                "Accepted_Groups": accepted_groups,
                "Rejected_Groups": rejected_groups,
                "Acceptance_Rate": acceptance_rate,
            }
        )
    return rows


def write_comparison_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["Acceptance_Rate"] = f"{row['Acceptance_Rate']:.6f}"
            writer.writerow(output_row)


def print_comparison_table(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    header = "Threshold | Accepted | Rejected | Acceptance Rate"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['Threshold']:<9.1f} | "
            f"{row['Accepted_Groups']:<8d} | "
            f"{row['Rejected_Groups']:<8d} | "
            f"{row['Acceptance_Rate']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare thresholds 0.4, 0.5, and 0.6 on real fusion scores."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing existing *_result.csv sensor outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Comparison CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_scores = load_existing_fusion_scores(args.input_root.resolve())
    comparison_rows = compare_thresholds(fusion_scores)
    write_comparison_csv(comparison_rows, args.output.resolve())
    print_comparison_table(comparison_rows)
    print(f"\nCSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
