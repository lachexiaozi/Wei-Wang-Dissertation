"""Fixed-weight baseline recorder for the dissertation experiments.

This module deliberately excludes adaptive sensor control.  CSI, RGB, and
LiDAR always run their independent detectors and always participate in object
fusion with equal weights (1/3 each).  Camera quality is still measured and
saved as diagnostic experiment data, but it cannot change participation or
weights in this baseline.

All geometry and decision mechanics are shared with the adaptive experiment:
calibrated bearing conversion, the 1-degree association gate, binary weighted
voting, and the 0.5 final fusion threshold.
"""

import time
from pathlib import Path

import cv2

from adaptive_object_fusion import fixed_weight_fusion
from camera_adaptive_control import get_camera_references
from detection_results import (
    build_detection_record,
    camera_detections_from_image,
    lidar_detections_from_scan,
    load_camera_detector,
)
from features_extract import calculate_camera_quality
from sensor_data import (
    FUSION_THRESHOLD,
    CSI_Camera_Data,
    LidarDisplay,
    RGB_Camera_Data,
    SensorBatchRecorder as AdaptiveSensorBatchRecorder,
    build_adaptive_fusion_record,
    build_final_obstacle_records,
    initialize_adaptive_fusion_results_csv,
    initialize_camera_features_csv,
    initialize_detection_results_csv,
    initialize_final_obstacle_results_csv,
)


FIXED_SENSOR_WEIGHT = 1.0 / 3.0
FIXED_WEIGHT_EXPERIMENT_DATA_DIRECTORY = (
    Path(__file__).resolve().parent / "Fixed Weight Experimental Data"
)


def _fixed_sensor_status(quality):
    """Attach fixed baseline metadata without applying adaptive control."""
    return {
        **quality,
        "trend": "Not Used",
        "trend_slope": None,
        "state": "Fixed",
        "active": True,
        "weight": FIXED_SENSOR_WEIGHT,
        "weight_label": "Fixed Weight",
    }


def build_fixed_monitoring_result(csi_quality, rgb_quality):
    """Build CSV/display metadata; these values never control the baseline."""
    return {
        "csi": _fixed_sensor_status(csi_quality),
        "rgb": _fixed_sensor_status(rgb_quality),
        "lidar": _fixed_sensor_status({"quality_score": 1.0}),
        "no_active_sensor": False,
    }


class SensorBatchRecorder(AdaptiveSensorBatchRecorder):
    """Record the all-sensor, equal-weight fixed fusion baseline."""

    @staticmethod
    def _create_experiment_directory(scenario_name, layout_id=None):
        """Reserve the next fixed-baseline test directory for one scenario."""
        scenario_name = str(scenario_name).strip()
        if not scenario_name or scenario_name in {".", ".."}:
            raise ValueError("scenario_name must be a non-empty folder name")
        if Path(scenario_name).name != scenario_name:
            raise ValueError("scenario_name cannot contain path separators")

        scenario_directory = (
            FIXED_WEIGHT_EXPERIMENT_DATA_DIRECTORY / scenario_name
        )
        scenario_directory.mkdir(parents=True, exist_ok=True)
        layout_suffix = "" if layout_id is None else f"_layout{int(layout_id)}"
        test_number = 1
        while True:
            legacy_directory = scenario_directory / f"test{test_number}"
            if legacy_directory.exists() or any(
                scenario_directory.glob(f"test{test_number}[_]*")
            ):
                test_number += 1
                continue
            experiment_directory = scenario_directory / (
                f"test{test_number}{layout_suffix}"
            )
            try:
                experiment_directory.mkdir(exist_ok=False)
                return experiment_directory
            except FileExistsError:
                test_number += 1

    def __init__(
        self,
        save_interval=0.2,
        sample_points=400,
        square_size=100,
        scenario_name="unknown_scenario",
        layout_id=None,
        max_frames=None,
    ):
        # This mirrors the adaptive recorder's experiment structure, while
        # intentionally creating no CameraAdaptiveController.
        self.save_interval = save_interval
        self.sample_points = sample_points
        self.next_sample_number = 1
        self.last_batch_time = None
        self.run_start_time = None
        self.frame_processing_times_ms = []
        self.detection_records = []
        self.adaptive_fusion_records = []
        self.final_obstacle_records = []
        self.max_frames = max_frames
        self.layout_id = int(layout_id) if layout_id is not None else None
        self.camera_references = get_camera_references(self.layout_id)
        self.experiment_directory = self._create_experiment_directory(
            scenario_name, layout_id
        )

        self.csi_directory = self.experiment_directory / "Front CSI Camera Data"
        self.rgb_directory = self.experiment_directory / "RGB Camera Data"
        self.lidar_directory = self.experiment_directory / "LiDAR Data"
        self.detection_directory = (
            self.experiment_directory / "Fixed Detection Results Data"
        )
        self.fusion_directory = (
            self.experiment_directory / "Fixed Fusion Results Data"
        )
        self.final_obstacle_directory = (
            self.experiment_directory / "Fixed Final Obstacle Results Data"
        )

        scenario_folder_name = self.experiment_directory.parent.name
        scenario_prefix, separator, weather_name = scenario_folder_name.partition("_")
        if not (
            separator
            and scenario_prefix.startswith("S")
            and scenario_prefix[1:].isdigit()
        ):
            weather_name = scenario_folder_name
        test_number = (
            self.experiment_directory.name.removeprefix("test").split("_", 1)[0]
        )
        layout_name = (
            f"layout{self.layout_id}"
            if self.layout_id is not None
            else "layout_unknown"
        )

        self.detection_results_path = self.detection_directory / (
            f"{weather_name}_test{test_number}_{layout_name}_fixed_result.csv"
        )
        # The inherited save/close methods use this established attribute name.
        self.adaptive_fusion_results_path = self.fusion_directory / (
            f"{weather_name}_test{test_number}_{layout_name}_fixed_fusion.csv"
        )
        self.final_obstacle_results_path = self.final_obstacle_directory / (
            f"{weather_name}_test{test_number}_{layout_name}_fixed_final_obstacles.csv"
        )
        self.camera_features_path = (
            self.experiment_directory
            / "Fixed Camera Features Data"
            / f"{weather_name}_T{test_number}_fixed.csv"
        )

        for directory in (
            self.csi_directory,
            self.rgb_directory,
            self.lidar_directory,
            self.detection_directory,
            self.fusion_directory,
            self.final_obstacle_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.lidar_display = LidarDisplay(square_size=square_size)
        initialize_camera_features_csv(self.camera_features_path)
        initialize_detection_results_csv(self.detection_results_path)
        initialize_adaptive_fusion_results_csv(
            self.adaptive_fusion_results_path
        )
        initialize_final_obstacle_results_csv(
            self.final_obstacle_results_path
        )
        self.camera_detector = load_camera_detector()

        print(f"Saving fixed-weight run to: {self.experiment_directory}")
        print(f"Camera quality CSV: {self.camera_features_path.name}")
        print(f"Fixed per-sensor detection CSV: {self.detection_results_path.name}")
        print(f"Fixed fusion CSV: {self.adaptive_fusion_results_path.name}")
        print(f"Fixed final obstacle CSV: {self.final_obstacle_results_path.name}")

    def update(self, qcar, weather="unknown"):
        """Acquire and process one due frame using the fixed baseline."""
        current_time = time.monotonic()
        if (
            self.last_batch_time is not None
            and current_time - self.last_batch_time < self.save_interval
        ):
            self.lidar_display.process_events()
            return (cv2.waitKey(1) & 0xFF) != 27

        self.last_batch_time = current_time
        if self.run_start_time is None:
            self.run_start_time = current_time
        timestamp = current_time - self.run_start_time
        frame_processing_start = time.perf_counter()

        lidar_success, angles, distances = qcar.get_lidar(
            samplePoints=self.sample_points
        )
        rgb_success, rgb_image = qcar.get_image(camera=qcar.CAMERA_RGB)
        csi_success, csi_image = qcar.get_image(camera=qcar.CAMERA_CSI_FRONT)

        quality_monitoring_start = time.perf_counter()
        rgb_quality = None
        csi_quality = None
        if rgb_success and rgb_image is not None:
            rgb_quality = calculate_camera_quality(
                rgb_image,
                self.camera_references["RGB"]["sharpness"],
                self.camera_references["RGB"]["contrast"],
            )
        if csi_success and csi_image is not None:
            csi_quality = calculate_camera_quality(
                csi_image,
                self.camera_references["CSI"]["sharpness"],
                self.camera_references["CSI"]["contrast"],
            )

        sensor_data_complete = all(
            (
                lidar_success,
                angles is not None,
                distances is not None,
                rgb_success,
                rgb_image is not None,
                csi_success,
                csi_image is not None,
                csi_quality is not None,
                rgb_quality is not None,
            )
        )

        if sensor_data_complete:
            fixed_status = build_fixed_monitoring_result(
                csi_quality, rgb_quality
            )
            quality_monitoring_ms = (
                time.perf_counter() - quality_monitoring_start
            ) * 1000.0

            self.lidar_display.update(
                angles, distances, fixed_status["lidar"]
            )
            RGB_Camera_Data(rgb_image, fixed_status["rgb"])
            CSI_Camera_Data(csi_image, fixed_status["csi"])

            # No Active/Inactive gate exists in the fixed baseline.  Every
            # independent detector executes for every complete frame.
            csi_detection_start = time.perf_counter()
            csi_detections = camera_detections_from_image(
                self.camera_detector, csi_image
            )
            csi_detection_ms = (
                time.perf_counter() - csi_detection_start
            ) * 1000.0

            rgb_detection_start = time.perf_counter()
            rgb_detections = camera_detections_from_image(
                self.camera_detector, rgb_image
            )
            rgb_detection_ms = (
                time.perf_counter() - rgb_detection_start
            ) * 1000.0

            lidar_detection_start = time.perf_counter()
            lidar_detections = lidar_detections_from_scan(angles, distances)
            lidar_detection_ms = (
                time.perf_counter() - lidar_detection_start
            ) * 1000.0

            frame = {
                "frame_id": self.next_sample_number,
                "timestamp": timestamp,
                "scenario": self.experiment_directory.parent.name,
                "weather": str(weather),
                "layout": self.layout_id,
            }
            timings = {
                "Frame_Processing_Time_ms": 0.0,
                # Kept in the common CSV column for direct comparison.  In
                # this file it is monitoring time, not control-decision time.
                "Quality_Control_Time_ms": quality_monitoring_ms,
                "YOLO_CSI_Time_ms": csi_detection_ms,
                "YOLO_RGB_Time_ms": rgb_detection_ms,
                "LiDAR_Detection_Time_ms": lidar_detection_ms,
            }
            detection_outputs = {
                "csi": csi_detections,
                "rgb": rgb_detections,
                "lidar": lidar_detections,
            }
            detection_record = build_detection_record(
                frame,
                fixed_status,
                detection_outputs,
                timings,
            )

            fusion_start = time.perf_counter()
            fusion_result = fixed_weight_fusion(
                frame_id=self.next_sample_number,
                detection_outputs=detection_outputs,
                fusion_threshold=FUSION_THRESHOLD,
            )
            fusion_time_ms = (
                time.perf_counter() - fusion_start
            ) * 1000.0

            # Add monitoring-only fields expected by the shared CSV schema.
            # None of them were inputs to fixed_weight_fusion().
            fusion_result.update(
                {
                    "CSI_q": float(csi_quality["quality_score"]),
                    "RGB_q": float(rgb_quality["quality_score"]),
                    "LiDAR_q": 1.0,
                    "CSI_State": "Fixed",
                    "RGB_State": "Fixed",
                    "LiDAR_State": "Fixed",
                    "CSI_Active": True,
                    "RGB_Active": True,
                    "LiDAR_Active": True,
                }
            )
            fusion_record = build_adaptive_fusion_record(
                frame,
                fusion_result,
                fusion_time_ms,
            )
            final_obstacle_records = build_final_obstacle_records(
                frame,
                fusion_result,
                fusion_time_ms,
            )

            self._save_complete_batch(
                angles,
                distances,
                rgb_image,
                csi_image,
                timestamp,
                weather,
                fixed_status,
                detection_record,
                fusion_record,
                final_obstacle_records,
            )
            frame_processing_ms = (
                time.perf_counter() - frame_processing_start
            ) * 1000.0
            detection_record["Frame_Processing_Time_ms"] = frame_processing_ms
            fusion_record["Frame_Processing_Time_ms"] = frame_processing_ms
            for record in final_obstacle_records:
                record["Frame_Processing_Time_ms"] = frame_processing_ms

            self.detection_records.append(detection_record)
            self.adaptive_fusion_records.append(fusion_record)
            self.final_obstacle_records.extend(final_obstacle_records)
            self.frame_processing_times_ms.append(frame_processing_ms)

            if (
                self.max_frames is not None
                and self.next_sample_number > self.max_frames
            ):
                print(f"Completed collection of {self.max_frames} fixed frames.")
                mean_processing_ms = sum(self.frame_processing_times_ms) / len(
                    self.frame_processing_times_ms
                )
                print(
                    f"Mean Frame_Processing_Time_ms: {mean_processing_ms:.6f}"
                )
                print(f"Mean processing FPS: {1000.0 / mean_processing_ms:.6f}")
                return False
        else:
            failed = []
            if not lidar_success or angles is None or distances is None:
                failed.append("LiDAR")
            if not rgb_success or rgb_image is None:
                failed.append("RGB")
            if not csi_success or csi_image is None:
                failed.append("Front CSI")
            print(f"Warning: skipped incomplete sensor batch ({', '.join(failed)})")

        return (cv2.waitKey(1) & 0xFF) != 27


FixedWeightSensorBatchRecorder = SensorBatchRecorder
