"""Per-sensor detector execution and synchronized result serialization.

This module deliberately records each sensor's detections independently.  It
does not aggregate detection scores or perform cross-sensor fusion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lidar_obstacle_detector import detect_obstacles


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = (
    ROOT
    / "YOLO_Obstacle_Training"
    / "yolo26s_obstacle_v1"
    / "weights"
    / "best.pt"
)

CAMERA_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640
SENSORS = ("csi", "rgb", "lidar")

DETECTION_RESULT_FIELDS = [
    "Frame_ID",
    "Timestamp",
    "Scenario",
    "Weather",
    "Layout_ID",
    "CSI_Quality",
    "CSI_Trend",
    "CSI_State",
    "CSI_Active",
    "CSI_Weight",
    "CSI_Detection_Count",
    "RGB_Quality",
    "RGB_Trend",
    "RGB_State",
    "RGB_Active",
    "RGB_Weight",
    "RGB_Detection_Count",
    "LiDAR_Quality",
    "LiDAR_Trend",
    "LiDAR_State",
    "LiDAR_Active",
    "LiDAR_Weight",
    "LiDAR_Detection_Count",
    "CSI_Detection_Scores",
    "RGB_Detection_Scores",
    "LiDAR_Detection_Scores",
    "CSI_Detections_JSON",
    "RGB_Detections_JSON",
    "LiDAR_Detections_JSON",
    "Frame_Processing_Time_ms",
    "Quality_Control_Time_ms",
    "YOLO_CSI_Time_ms",
    "YOLO_RGB_Time_ms",
    "LiDAR_Detection_Time_ms",
]


def _camera_detections_from_result(result: Any) -> list[dict[str, Any]]:
    """Serialize bounding boxes from one YOLO inference."""
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    return [
        {
            "bbox": [float(coordinate) for coordinate in box],
            "confidence": float(confidence),
        }
        for box, confidence in zip(boxes, confidences)
    ]


def camera_detections_from_image(
    model: Any, image: np.ndarray
) -> list[dict[str, Any]]:
    """Run the CSI/RGB YOLO detector once and serialize its full result."""
    result = model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=CAMERA_CONFIDENCE,
        device="cpu",
        verbose=False,
    )[0]
    return _camera_detections_from_result(result)


def _lidar_detections_from_result(result: Any) -> list[dict[str, Any]]:
    """Serialize clusters from one LiDAR Detector V1 call."""
    return [
        {
            "cluster_id": int(cluster.cluster_id),
            "point_count": int(cluster.point_count),
            "centroid_x": float(cluster.centroid_x),
            "centroid_y": float(cluster.centroid_y),
            "min_distance": float(cluster.min_distance),
            "LiDAR_detection_score": float(cluster.LiDAR_detection_score),
        }
        for cluster in result.clusters
    ]


def lidar_detections_from_scan(
    angles: np.ndarray, distances: np.ndarray
) -> list[dict[str, Any]]:
    """Run LiDAR Detector V1 once and serialize all detected clusters."""
    return _lidar_detections_from_result(detect_obstacles(angles, distances))


def load_camera_detector() -> Any:
    """Load the frozen YOLO model once before entering the frame loop."""
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Frozen YOLO weights not found: {WEIGHTS_PATH}")
    config_directory = ROOT / "YOLO_Obstacle_Training" / ".ultralytics_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_directory))
    from ultralytics import YOLO

    return YOLO(str(WEIGHTS_PATH))


def build_detection_record(
    frame: dict[str, Any],
    controller_result: dict[str, Any],
    detection_outputs: dict[str, Sequence[dict[str, Any]]],
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build one synchronized record without combining sensor detections."""
    missing = set(SENSORS) - detection_outputs.keys()
    if missing:
        raise KeyError(f"Missing detector outputs: {sorted(missing)}")

    record: dict[str, Any] = {
        "Frame_ID": frame["frame_id"],
        "Timestamp": frame["timestamp"],
        "Scenario": frame["scenario"],
        "Weather": frame["weather"],
        "Layout_ID": frame["layout"],
    }
    labels = {"csi": "CSI", "rgb": "RGB", "lidar": "LiDAR"}
    for sensor in SENSORS:
        label = labels[sensor]
        control = controller_result[sensor]
        detections = list(detection_outputs[sensor])
        score_key = "LiDAR_detection_score" if sensor == "lidar" else "confidence"
        scores = [float(detection[score_key]) for detection in detections]
        record.update(
            {
                f"{label}_Quality": float(control["quality_score"]),
                f"{label}_Trend": control["trend"],
                f"{label}_State": control["state"],
                f"{label}_Active": bool(control["active"]),
                f"{label}_Weight": float(control["weight"]),
                f"{label}_Detection_Count": len(detections),
                f"{label}_Detection_Scores": json.dumps(
                    scores, separators=(",", ":")
                ),
                f"{label}_Detections_JSON": json.dumps(
                    detections, separators=(",", ":")
                ),
            }
        )

    default_timings = {
        "Frame_Processing_Time_ms": 0.0,
        "Quality_Control_Time_ms": 0.0,
        "YOLO_CSI_Time_ms": 0.0,
        "YOLO_RGB_Time_ms": 0.0,
        "LiDAR_Detection_Time_ms": 0.0,
    }
    if timings is not None:
        default_timings.update(timings)
    record.update(default_timings)
    if set(record) != set(DETECTION_RESULT_FIELDS):
        raise AssertionError("Detection record does not match the CSV schema")
    return record
