"""Independent camera-LiDAR bearing-angle calibration recorder.

This script records every CSI, RGB, and LiDAR detection independently.  It
deliberately performs no cross-sensor association, object matching, voting,
fusion, ground-truth lookup, or final detection decision.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from detection_results import (
    camera_detections_from_image,
    lidar_detections_from_scan,
    load_camera_detector,
)
from obstacle_test import spawn_obstacle_test_scene
from qvl.qlabs import QuanserInteractiveLabs
from sensor_data import LidarDisplay


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT / "bearing_calibration_results.csv"

CSI_WIDTH = 820
CSI_HFOV_DEG = 160.0
CSI_OFFSET_DEG = 0.0

RGB_WIDTH = 640
RGB_HFOV_DEG = 69.0
RGB_OFFSET_DEG = 0.0

CSI_WINDOW = "Bearing Calibration - CSI"
RGB_WINDOW = "Bearing Calibration - RGB"

CSV_FIELDS = (
    "Frame_ID",
    "Sensor",
    "Detection_ID",
    "cluster_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "center_x",
    "centroid_x",
    "centroid_y",
    "point_count",
    "min_distance",
    "detection_score",
    "bearing_deg",
)


def camera_bearing_deg(
    bbox: Iterable[float],
    image_width: int,
    horizontal_fov_deg: float,
    offset_deg: float = 0.0,
) -> tuple[float, float]:
    """Return bbox horizontal centre and its approximate camera bearing."""
    coordinates = [float(value) for value in bbox]
    if len(coordinates) != 4:
        raise ValueError(f"Camera bbox must have four coordinates: {coordinates}")
    x1, _, x2, _ = coordinates
    center_x = (x1 + x2) / 2.0
    bearing = (
        ((center_x - image_width / 2.0) / (image_width / 2.0))
        * (horizontal_fov_deg / 2.0)
        + offset_deg
    )
    return center_x, bearing


def lidar_bearing_deg(centroid_x: float, centroid_y: float) -> float:
    """Return LiDAR bearing without changing the detector coordinate signs."""
    bearing_rad = math.atan2(float(centroid_x), float(centroid_y))
    return math.degrees(bearing_rad)


def _validate_image_width(image: np.ndarray, sensor: str, expected_width: int) -> None:
    if image is None or image.ndim < 2:
        raise ValueError(f"{sensor} returned an invalid image")
    actual_width = int(image.shape[1])
    if actual_width != expected_width:
        raise RuntimeError(
            f"{sensor} image width is {actual_width}, expected {expected_width}. "
            "Bearing recording stopped to avoid using an incorrect calibration width."
        )


def _camera_measurements(
    frame_id: int,
    sensor: str,
    detections: Iterable[dict[str, Any]],
    image_width: int,
    horizontal_fov_deg: float,
    offset_deg: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurements = []
    rows = []
    for detection_id, detection in enumerate(detections, start=1):
        bbox = [float(value) for value in detection["bbox"]]
        confidence = float(detection["confidence"])
        center_x, bearing = camera_bearing_deg(
            bbox,
            image_width,
            horizontal_fov_deg,
            offset_deg,
        )
        measurement = {
            "detection_id": detection_id,
            "bbox": bbox,
            "confidence": confidence,
            "center_x": center_x,
            "bearing_deg": bearing,
        }
        measurements.append(measurement)
        rows.append(
            {
                "Frame_ID": frame_id,
                "Sensor": sensor,
                "Detection_ID": detection_id,
                "cluster_id": "",
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "confidence": confidence,
                "center_x": center_x,
                "centroid_x": "",
                "centroid_y": "",
                "point_count": "",
                "min_distance": "",
                "detection_score": "",
                "bearing_deg": bearing,
            }
        )
    return measurements, rows


def _lidar_measurements(
    frame_id: int,
    detections: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurements = []
    rows = []
    for detection in detections:
        cluster_id = int(detection["cluster_id"])
        centroid_x = float(detection["centroid_x"])
        centroid_y = float(detection["centroid_y"])
        point_count = int(detection["point_count"])
        min_distance = float(detection["min_distance"])
        detection_score = float(detection["LiDAR_detection_score"])
        bearing = lidar_bearing_deg(centroid_x, centroid_y)
        measurement = {
            "cluster_id": cluster_id,
            "point_count": point_count,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "min_distance": min_distance,
            "detection_score": detection_score,
            "bearing_deg": bearing,
        }
        measurements.append(measurement)
        rows.append(
            {
                "Frame_ID": frame_id,
                "Sensor": "LiDAR",
                "Detection_ID": cluster_id,
                "cluster_id": cluster_id,
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "confidence": "",
                "center_x": "",
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "point_count": point_count,
                "min_distance": min_distance,
                "detection_score": detection_score,
                "bearing_deg": bearing,
            }
        )
    return measurements, rows


def _format_bbox(bbox: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.1f}" for value in bbox) + "]"


def print_frame_measurements(
    frame_id: int,
    csi_measurements: list[dict[str, Any]],
    rgb_measurements: list[dict[str, Any]],
    lidar_measurements: list[dict[str, Any]],
) -> None:
    print(f"\n===== Frame {frame_id} =====")
    for sensor, measurements in (
        ("CSI", csi_measurements),
        ("RGB", rgb_measurements),
    ):
        print(f"\n{sensor}:")
        if not measurements:
            print("  No detections")
        for measurement in measurements:
            print(
                f"  Det {measurement['detection_id']} | "
                f"bbox={_format_bbox(measurement['bbox'])} | "
                f"conf={measurement['confidence']:.3f} | "
                f"center_x={measurement['center_x']:.2f} | "
                f"bearing={measurement['bearing_deg']:.2f} deg"
            )

    print("\nLiDAR:")
    if not lidar_measurements:
        print("  No detections")
    for measurement in lidar_measurements:
        print(
            f"  Cluster {measurement['cluster_id']} | "
            f"centroid=({measurement['centroid_x']:.2f}, "
            f"{measurement['centroid_y']:.2f}) | "
            f"points={measurement['point_count']} | "
            f"min_distance={measurement['min_distance']:.2f} m | "
            f"score={measurement['detection_score']:.2f} | "
            f"bearing={measurement['bearing_deg']:.2f} deg"
        )


def draw_camera_measurements(
    image: np.ndarray,
    measurements: Iterable[dict[str, Any]],
) -> np.ndarray:
    """Draw detector boxes, confidence, and bearing on a copy of an image."""
    display = np.asarray(image).copy()
    for measurement in measurements:
        x1, y1, x2, y2 = (
            int(round(value)) for value in measurement["bbox"]
        )
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (
            f"conf={measurement['confidence']:.2f}  "
            f"bearing={measurement['bearing_deg']:.2f} deg"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        text_y = max(text_height + baseline + 2, y1 - 7)
        cv2.rectangle(
            display,
            (x1, text_y - text_height - baseline - 3),
            (x1 + text_width + 4, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            display,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return display


def _unused_csv_row_fields_are_present(row: dict[str, Any]) -> None:
    if set(row) != set(CSV_FIELDS):
        raise AssertionError("Bearing calibration row does not match CSV schema")


def _choose_output_path(requested_path: Path) -> Path:
    """Choose the base CSV name, followed by fixed run numbers 2 through 6."""
    requested_path = requested_path.expanduser().resolve()
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    if not requested_path.exists():
        return requested_path

    for run_number in range(2, 7):
        candidate = requested_path.with_name(
            f"{requested_path.stem}_{run_number}{requested_path.suffix}"
        )
        if not candidate.exists():
            return candidate

    raise FileExistsError(
        f"Calibration CSV slots 1-6 already exist in {requested_path.parent}. "
        "Move or rename an existing result before starting another run."
    )


def run_calibration(args: argparse.Namespace) -> Path:
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open(args.host):
        raise RuntimeError(f"Unable to connect to QLabs at {args.host!r}")

    lidar_display = None
    output_path = None
    try:
        print("Creating the four-QCar scene from obstacle_test.py...")
        qcars = spawn_obstacle_test_scene(qlabs, destroy_existing=True)
        qcar = qcars["Qcar1"]
        print("Loading the existing camera detector model...")
        camera_detector = load_camera_detector()

        output_path = _choose_output_path(args.csv)
        cv2.startWindowThread()
        lidar_display = LidarDisplay(square_size=args.lidar_display_range)

        print(f"Calibration CSV: {output_path}")
        print("Press Esc or q in a camera window, or Ctrl+C in the terminal, to stop.")
        frame_id = 0

        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            csv_file.flush()

            while args.max_frames is None or frame_id < args.max_frames:
                frame_start = time.monotonic()
                lidar_ok, angles, distances = qcar.get_lidar(
                    samplePoints=args.sample_points
                )
                rgb_ok, rgb_image = qcar.get_image(camera=qcar.CAMERA_RGB)
                csi_ok, csi_image = qcar.get_image(camera=qcar.CAMERA_CSI_FRONT)

                if not all(
                    (
                        lidar_ok,
                        angles is not None,
                        distances is not None,
                        rgb_ok,
                        rgb_image is not None,
                        csi_ok,
                        csi_image is not None,
                    )
                ):
                    failed = []
                    if not lidar_ok or angles is None or distances is None:
                        failed.append("LiDAR")
                    if not rgb_ok or rgb_image is None:
                        failed.append("RGB")
                    if not csi_ok or csi_image is None:
                        failed.append("CSI")
                    print(f"Warning: skipped incomplete acquisition ({', '.join(failed)})")
                else:
                    _validate_image_width(csi_image, "CSI", CSI_WIDTH)
                    _validate_image_width(rgb_image, "RGB", RGB_WIDTH)
                    frame_id += 1

                    csi_detections = camera_detections_from_image(
                        camera_detector, csi_image
                    )
                    rgb_detections = camera_detections_from_image(
                        camera_detector, rgb_image
                    )
                    lidar_detections = lidar_detections_from_scan(
                        angles, distances
                    )

                    csi_measurements, csi_rows = _camera_measurements(
                        frame_id,
                        "CSI",
                        csi_detections,
                        CSI_WIDTH,
                        CSI_HFOV_DEG,
                        CSI_OFFSET_DEG,
                    )
                    rgb_measurements, rgb_rows = _camera_measurements(
                        frame_id,
                        "RGB",
                        rgb_detections,
                        RGB_WIDTH,
                        RGB_HFOV_DEG,
                        RGB_OFFSET_DEG,
                    )
                    lidar_measurements, lidar_rows = _lidar_measurements(
                        frame_id, lidar_detections
                    )

                    rows = [*csi_rows, *rgb_rows, *lidar_rows]
                    for row in rows:
                        _unused_csv_row_fields_are_present(row)
                    writer.writerows(rows)
                    csv_file.flush()

                    print_frame_measurements(
                        frame_id,
                        csi_measurements,
                        rgb_measurements,
                        lidar_measurements,
                    )
                    cv2.imshow(
                        CSI_WINDOW,
                        draw_camera_measurements(csi_image, csi_measurements),
                    )
                    cv2.imshow(
                        RGB_WINDOW,
                        draw_camera_measurements(rgb_image, rgb_measurements),
                    )
                    lidar_display.update(angles, distances)

                elapsed = time.monotonic() - frame_start
                wait_ms = max(1, int(max(0.0, args.interval - elapsed) * 1000.0))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q")):
                    break

        return output_path
    finally:
        if lidar_display is not None:
            lidar_display.close()
        cv2.destroyAllWindows()
        qlabs.close()
        if output_path is not None:
            print(f"Calibration data saved to: {output_path}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the obstacle_test.py scene and record independent CSI, "
            "RGB, and LiDAR bearing detections."
        )
    )
    parser.add_argument("--host", default="localhost", help="QLabs host")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV output path (default: {DEFAULT_CSV_PATH.name})",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="stop after this many complete frames (default: run until stopped)",
    )
    parser.add_argument(
        "--interval",
        type=nonnegative_float,
        default=0.2,
        help="minimum seconds between frame starts (default: 0.2)",
    )
    parser.add_argument(
        "--sample-points",
        type=positive_int,
        default=400,
        help="LiDAR samples per scan (default: 400)",
    )
    parser.add_argument(
        "--lidar-display-range",
        type=positive_int,
        default=40,
        help="LiDAR display half-range in metres (default: 40)",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        run_calibration(args)
    except KeyboardInterrupt:
        print("\nCalibration stopped by Ctrl+C.")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
