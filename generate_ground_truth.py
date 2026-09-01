"""Generate static obstacle Ground Truth from the three QLabs layouts.

Ground Truth is an offline evaluation input only. It must never be imported by
the online quality, sensor-selection, detection, association, or fusion path.
"""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import obstacle_layout_1
import obstacle_layout_2
import obstacle_layout_3


ROOT = Path(__file__).resolve().parent
QCAR2_PATH = ROOT / "qcar2.py"
OUTPUT_DIRECTORY = ROOT / "Ground Truth Data"
OUTPUT_PATH = OUTPUT_DIRECTORY / "ground_truth_obstacles.csv"

LAYOUTS = (obstacle_layout_1, obstacle_layout_2, obstacle_layout_3)

GROUND_TRUTH_FIELDS = (
    "Layout_ID",
    "Layout_Name",
    "Ground_Truth_ID",
    "Object_Type",
    "Object_Name",
    "Actor_ID",
    "World_X_m",
    "World_Y_m",
    "World_Z_m",
    "Yaw_deg",
    "Scale_X",
    "Scale_Y",
    "Scale_Z",
    "Configuration",
    "Ego_X_m",
    "Ego_Y_m",
    "Ego_Z_m",
    "Forward_Distance_m",
    "Sensor_Lateral_Distance_m",
    "Vertical_Offset_m",
    "Ground_Truth_Range_m",
    "Ground_Truth_Bearing_deg",
    "In_Front",
    "Target_For_Evaluation",
    "Source_File",
)


def load_ego_qcar_location(qcar2_path: Path = QCAR2_PATH) -> tuple[float, ...]:
    """Read the literal ego location without importing QLabs dependencies."""
    tree = ast.parse(qcar2_path.read_text(encoding="utf-8"), filename=str(qcar2_path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "EGO_QCAR_LOCATION"
            for target in targets
        ):
            continue
        value_node = node.value
        location = ast.literal_eval(value_node)
        if not isinstance(location, (list, tuple)) or len(location) != 3:
            raise ValueError("EGO_QCAR_LOCATION must contain exactly x, y, z")
        parsed = tuple(float(value) for value in location)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError("EGO_QCAR_LOCATION must contain finite numbers")
        return parsed
    raise ValueError("qcar2.py does not define EGO_QCAR_LOCATION")


def _finite_triplet(values: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    parsed = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field_name} must contain finite values")
    return parsed


def _ground_truth_row(
    layout: ModuleType,
    ground_truth_number: int,
    object_type: str,
    object_name: str,
    actor_id: int | str,
    location: Any,
    ego_location: tuple[float, float, float],
    yaw_degrees: float = 0.0,
    scale: Any = (1.0, 1.0, 1.0),
    configuration: int | str = "",
) -> dict[str, Any]:
    world_x, world_y, world_z = _finite_triplet(location, "location")
    scale_x, scale_y, scale_z = _finite_triplet(scale, "scale")
    ego_x, ego_y, ego_z = ego_location

    # QCar1 faces +world-X. The QLabs LiDAR convention used by this project has
    # positive sensor-x towards decreasing world-Y, hence ego_y - world_y.
    forward_distance = world_x - ego_x
    sensor_lateral_distance = ego_y - world_y
    vertical_offset = world_z - ego_z
    horizontal_range = math.hypot(forward_distance, sensor_lateral_distance)
    bearing_degrees = math.degrees(
        math.atan2(sensor_lateral_distance, forward_distance)
    )

    return {
        "Layout_ID": int(layout.LAYOUT_ID),
        "Layout_Name": str(layout.LAYOUT_NAME),
        "Ground_Truth_ID": f"L{layout.LAYOUT_ID}_GT{ground_truth_number:02d}",
        "Object_Type": object_type,
        "Object_Name": object_name,
        "Actor_ID": actor_id,
        "World_X_m": world_x,
        "World_Y_m": world_y,
        "World_Z_m": world_z,
        "Yaw_deg": float(yaw_degrees),
        "Scale_X": scale_x,
        "Scale_Y": scale_y,
        "Scale_Z": scale_z,
        "Configuration": configuration,
        "Ego_X_m": ego_x,
        "Ego_Y_m": ego_y,
        "Ego_Z_m": ego_z,
        "Forward_Distance_m": round(forward_distance, 9),
        "Sensor_Lateral_Distance_m": round(sensor_lateral_distance, 9),
        "Vertical_Offset_m": round(vertical_offset, 9),
        "Ground_Truth_Range_m": round(horizontal_range, 9),
        "Ground_Truth_Bearing_deg": round(bearing_degrees, 9),
        "In_Front": forward_distance > 0.0,
        "Target_For_Evaluation": True,
        "Source_File": Path(layout.__file__).name,
    }


def build_ground_truth_rows() -> list[dict[str, Any]]:
    """Build all physical obstacle rows in layout-definition order."""
    ego_location = load_ego_qcar_location()
    rows: list[dict[str, Any]] = []

    for layout in LAYOUTS:
        ground_truth_number = 1

        for name, actor_id, location, yaw_degrees in layout.QCAR_OBSTACLES:
            rows.append(
                _ground_truth_row(
                    layout,
                    ground_truth_number,
                    "QCar",
                    name,
                    actor_id,
                    location,
                    ego_location,
                    yaw_degrees=yaw_degrees,
                )
            )
            ground_truth_number += 1

        cube_yaw = float(getattr(layout, "CUBE_ROTATION", [0, 0, 0])[2])
        for cube_number, (actor_id, location, scale) in enumerate(
            layout.CUBE_OBSTACLES,
            start=1,
        ):
            rows.append(
                _ground_truth_row(
                    layout,
                    ground_truth_number,
                    "Cube",
                    f"Cube{cube_number}",
                    actor_id,
                    location,
                    ego_location,
                    yaw_degrees=cube_yaw,
                    scale=scale,
                )
            )
            ground_truth_number += 1

        for cone_number, (location, rotation, scale, configuration) in enumerate(
            getattr(layout, "TRAFFIC_CONE_OBSTACLES", ()),
            start=1,
        ):
            rotation_values = _finite_triplet(rotation, "cone rotation")
            rows.append(
                _ground_truth_row(
                    layout,
                    ground_truth_number,
                    "Cone",
                    f"Cone{cone_number}",
                    "",
                    location,
                    ego_location,
                    yaw_degrees=rotation_values[2],
                    scale=scale,
                    configuration=configuration,
                )
            )
            ground_truth_number += 1

    return rows


def validate_ground_truth_rows(rows: list[dict[str, Any]]) -> None:
    """Reject incomplete or internally inconsistent generated Ground Truth."""
    expected_counts = {
        int(layout.LAYOUT_ID): (
            len(layout.QCAR_OBSTACLES)
            + len(layout.CUBE_OBSTACLES)
            + len(getattr(layout, "TRAFFIC_CONE_OBSTACLES", ()))
        )
        for layout in LAYOUTS
    }
    actual_counts = {
        layout_id: sum(row["Layout_ID"] == layout_id for row in rows)
        for layout_id in expected_counts
    }
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Ground Truth row counts disagree: {actual_counts} != {expected_counts}"
        )
    ground_truth_ids = [row["Ground_Truth_ID"] for row in rows]
    if len(ground_truth_ids) != len(set(ground_truth_ids)):
        raise AssertionError("Ground_Truth_ID values must be unique")
    if not all(row["In_Front"] for row in rows):
        raise AssertionError("Every configured dissertation obstacle must be in front")
    for row in rows:
        if set(row) != set(GROUND_TRUTH_FIELDS):
            raise AssertionError("Ground Truth row does not match the CSV schema")


def write_ground_truth_csv(
    rows: list[dict[str, Any]], output_path: Path = OUTPUT_PATH
) -> Path:
    """Write the validated master Ground Truth CSV."""
    validate_ground_truth_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GROUND_TRUTH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    rows = build_ground_truth_rows()
    output_path = write_ground_truth_csv(rows)
    print(f"Ground Truth CSV: {output_path}")
    for layout in LAYOUTS:
        layout_rows = [row for row in rows if row["Layout_ID"] == layout.LAYOUT_ID]
        print(f"Layout {layout.LAYOUT_ID}: {len(layout_rows)} obstacles")


if __name__ == "__main__":
    main()
