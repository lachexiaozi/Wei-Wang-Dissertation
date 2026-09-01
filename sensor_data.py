import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from adaptive_object_fusion import adaptive_object_fusion
from camera_adaptive_control import (
    CameraAdaptiveController,
    get_camera_references,
)
from features_extract import calculate_camera_quality
from detection_results import (
    DETECTION_RESULT_FIELDS,
    build_detection_record,
    camera_detections_from_image,
    lidar_detections_from_scan,
    load_camera_detector,
)


CSI_FRONT_WINDOW = "QCar1 Front CSI Camera"
RGB_WINDOW = "QCar1 RGB Camera"

STATUS_TREND_COLORS = {
    "Improving": (70, 220, 70),
    "Stable": (0, 215, 255),
    "Degrading": (70, 70, 255),
    "Initializing": (220, 220, 220),
}

CSI_DATA_DIRECTORY = Path(__file__).resolve().parent / "Front CSI Camera Data"
RGB_DATA_DIRECTORY = Path(__file__).resolve().parent / "RGB Camera Data"
LIDAR_DATA_DIRECTORY = Path(__file__).resolve().parent / "LiDAR Data"
EXPERIMENT_DATA_DIRECTORY = (
    Path(__file__).resolve().parent / "Dynamic Weight Experimental Data"
)
FUSION_THRESHOLD = 0.5
CAMERA_FEATURE_FIELDS = (
    "Frame_ID",
    "Timestamp",
    "Layout_ID",
    "Weather",
    "CSI_Sharpness",
    "CSI_Contrast",
    "CSI_Sharpness_Score",
    "CSI_Contrast_Score",
    "CSI_Quality_Score",
    "CSI_Trend",
    "CSI_Trend_Slope",
    "CSI_State",
    "CSI_Active",
    "CSI_Weight",
    "RGB_Sharpness",
    "RGB_Contrast",
    "RGB_Sharpness_Score",
    "RGB_Contrast_Score",
    "RGB_Quality_Score",
    "RGB_Trend",
    "RGB_Trend_Slope",
    "RGB_State",
    "RGB_Active",
    "RGB_Weight",
    "LiDAR_Quality_Score",
    "LiDAR_Trend",
    "LiDAR_Trend_Slope",
    "LiDAR_State",
    "LiDAR_Active",
    "LiDAR_Weight",
    "No_Active_Sensor",
)
ADAPTIVE_FUSION_FIELDS = (
    "Frame_ID",
    "Timestamp",
    "Scenario",
    "Weather",
    "Layout_ID",
    "CSI_Quality",
    "CSI_State",
    "CSI_Active",
    "CSI_Weight",
    "RGB_Quality",
    "RGB_State",
    "RGB_Active",
    "RGB_Weight",
    "LiDAR_Quality",
    "LiDAR_State",
    "LiDAR_Active",
    "LiDAR_Weight",
    "Active_Sensors",
    "Number_of_Object_Groups",
    "Number_of_Final_Obstacles",
    "No_Active_Sensor",
    "Bearing_Tolerance_deg",
    "Fusion_Threshold",
    "Fusion_Time_ms",
    "Frame_Processing_Time_ms",
    "Object_Groups_JSON",
)
FINAL_OBSTACLE_FIELDS = (
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
    "CSI_Participating",
    "RGB_Participating",
    "LiDAR_Participating",
    "CSI_Detected",
    "RGB_Detected",
    "LiDAR_Detected",
    "CSI_Weight",
    "RGB_Weight",
    "LiDAR_Weight",
    "CSI_Calibrated_Bearing",
    "RGB_Calibrated_Bearing",
    "LiDAR_Bearing",
    "CSI_RGB_Delta",
    "CSI_LiDAR_Delta",
    "RGB_LiDAR_Delta",
    "CSI_Detection_ID",
    "CSI_BBox_X1",
    "CSI_BBox_Y1",
    "CSI_BBox_X2",
    "CSI_BBox_Y2",
    "CSI_Confidence",
    "CSI_Center_X",
    "CSI_Raw_Bearing",
    "RGB_Detection_ID",
    "RGB_BBox_X1",
    "RGB_BBox_Y1",
    "RGB_BBox_X2",
    "RGB_BBox_Y2",
    "RGB_Confidence",
    "RGB_Center_X",
    "RGB_Raw_Bearing",
    "LiDAR_Cluster_ID",
    "LiDAR_Point_Count",
    "LiDAR_Centroid_X",
    "LiDAR_Centroid_Y",
    "LiDAR_Min_Distance",
    "LiDAR_Detection_Score",
    "Fusion_Time_ms",
    "Frame_Processing_Time_ms",
)


def initialize_camera_features_csv(csv_path):
    """Create a camera-feature CSV and write its header."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CAMERA_FEATURE_FIELDS)
        writer.writeheader()


def initialize_detection_results_csv(csv_path):
    """Create the synchronized per-sensor detection-results CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETECTION_RESULT_FIELDS)
        writer.writeheader()


def initialize_adaptive_fusion_results_csv(csv_path):
    """Create the per-frame adaptive object-fusion results CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ADAPTIVE_FUSION_FIELDS)
        writer.writeheader()


def initialize_final_obstacle_results_csv(csv_path):
    """Create the long-format accepted-obstacle results CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FINAL_OBSTACLE_FIELDS)
        writer.writeheader()


def save_frame_detection_results(csv_path, record):
    """Append one synchronized frame's independent detection results."""
    if set(record) != set(DETECTION_RESULT_FIELDS):
        raise AssertionError("Detection record does not match the CSV schema")
    with Path(csv_path).open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETECTION_RESULT_FIELDS)
        writer.writerow(record)


def rewrite_frame_detection_results(csv_path, records):
    """Finalize measured frame timings after their original CSV saves complete."""
    with Path(csv_path).open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETECTION_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def save_adaptive_fusion_result(csv_path, record):
    """Append one frame's adaptive association and binary-vote result."""
    if set(record) != set(ADAPTIVE_FUSION_FIELDS):
        raise AssertionError("Adaptive fusion record does not match the CSV schema")
    with Path(csv_path).open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ADAPTIVE_FUSION_FIELDS)
        writer.writerow(record)


def rewrite_adaptive_fusion_results(csv_path, records):
    """Persist finalized frame timings for adaptive fusion records."""
    with Path(csv_path).open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ADAPTIVE_FUSION_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def save_final_obstacle_results(csv_path, records):
    """Append every accepted obstacle in a frame as one flat CSV row."""
    for record in records:
        if set(record) != set(FINAL_OBSTACLE_FIELDS):
            raise AssertionError(
                "Final obstacle record does not match the CSV schema"
            )
    if not records:
        return
    with Path(csv_path).open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FINAL_OBSTACLE_FIELDS)
        writer.writerows(records)


def rewrite_final_obstacle_results(csv_path, records):
    """Persist finalized frame timings for accepted-obstacle rows."""
    with Path(csv_path).open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FINAL_OBSTACLE_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def build_adaptive_fusion_record(frame, fusion_result, fusion_time_ms):
    """Flatten one frame's candidate-group summary."""
    fusion_threshold = fusion_result["Fusion_Threshold"]
    if fusion_threshold is None:
        raise AssertionError("Final obstacle output requires a fusion threshold")
    number_of_final_obstacles = sum(
        group["Fusion_Decision"] is True
        for group in fusion_result["Object_Groups"]
    )
    record = {
        "Frame_ID": frame["frame_id"],
        "Timestamp": frame["timestamp"],
        "Scenario": frame["scenario"],
        "Weather": frame["weather"],
        "Layout_ID": frame["layout"],
        "CSI_Quality": fusion_result["CSI_q"],
        "CSI_State": fusion_result["CSI_State"],
        "CSI_Active": fusion_result["CSI_Active"],
        "CSI_Weight": fusion_result["CSI_Weight"],
        "RGB_Quality": fusion_result["RGB_q"],
        "RGB_State": fusion_result["RGB_State"],
        "RGB_Active": fusion_result["RGB_Active"],
        "RGB_Weight": fusion_result["RGB_Weight"],
        "LiDAR_Quality": fusion_result["LiDAR_q"],
        "LiDAR_State": fusion_result["LiDAR_State"],
        "LiDAR_Active": fusion_result["LiDAR_Active"],
        "LiDAR_Weight": fusion_result["LiDAR_Weight"],
        "Active_Sensors": "|".join(fusion_result["Active_Sensors"]),
        "Number_of_Object_Groups": fusion_result["Number_of_Object_Groups"],
        "Number_of_Final_Obstacles": number_of_final_obstacles,
        "No_Active_Sensor": fusion_result["No_Active_Sensor"],
        "Bearing_Tolerance_deg": fusion_result["Bearing_Tolerance_deg"],
        "Fusion_Threshold": "" if fusion_threshold is None else fusion_threshold,
        "Fusion_Time_ms": fusion_time_ms,
        "Frame_Processing_Time_ms": 0.0,
        "Object_Groups_JSON": fusion_result["Object_Groups_JSON"],
    }
    if set(record) != set(ADAPTIVE_FUSION_FIELDS):
        raise AssertionError("Adaptive fusion record does not match the CSV schema")
    return record


def build_final_obstacle_records(frame, fusion_result, fusion_time_ms):
    """Return one flat, plotting-friendly row per accepted object group."""
    threshold = fusion_result["Fusion_Threshold"]
    if threshold is None:
        raise AssertionError("Final obstacle output requires a fusion threshold")

    def value(member, key):
        return "" if member is None else member.get(key, "")

    def bbox_value(member, index):
        if member is None:
            return ""
        bbox = member.get("bbox")
        return "" if bbox is None else bbox[index]

    records = []
    accepted_groups = (
        group
        for group in fusion_result["Object_Groups"]
        if group["Fusion_Decision"] is True
    )
    for final_obstacle_id, group in enumerate(accepted_groups, start=1):
        members = group["members"]
        supporting_sensors = [
            sensor
            for sensor in ("CSI", "RGB", "LiDAR")
            if group["sensor_presence"][sensor]
        ]
        record = {
            "Frame_ID": frame["frame_id"],
            "Final_Obstacle_ID": final_obstacle_id,
            "Source_Group_ID": group["Group_ID"],
            "Timestamp": frame["timestamp"],
            "Scenario": frame["scenario"],
            "Weather": frame["weather"],
            "Layout_ID": frame["layout"],
            "Representative_Bearing_deg": group["Representative_Bearing_deg"],
            "Fusion_Score": group["Fusion_Score"],
            "Fusion_Threshold": threshold,
            "Fusion_Decision": group["Fusion_Decision"],
            "Active_Sensors": "|".join(group["active_sensors"]),
            "Supporting_Sensors": "|".join(supporting_sensors),
            "CSI_Participating": group["CSI_Participating"],
            "RGB_Participating": group["RGB_Participating"],
            "LiDAR_Participating": group["LiDAR_Participating"],
            "CSI_Detected": group["CSI_Detected"],
            "RGB_Detected": group["RGB_Detected"],
            "LiDAR_Detected": group["LiDAR_Detected"],
            "CSI_Weight": group["CSI_Weight"],
            "RGB_Weight": group["RGB_Weight"],
            "LiDAR_Weight": group["LiDAR_Weight"],
            "CSI_Calibrated_Bearing": group["CSI_Calibrated_Bearing"],
            "RGB_Calibrated_Bearing": group["RGB_Calibrated_Bearing"],
            "LiDAR_Bearing": group["LiDAR_Bearing"],
            "CSI_RGB_Delta": "" if group["CSI_RGB_Delta"] is None else group["CSI_RGB_Delta"],
            "CSI_LiDAR_Delta": "" if group["CSI_LiDAR_Delta"] is None else group["CSI_LiDAR_Delta"],
            "RGB_LiDAR_Delta": "" if group["RGB_LiDAR_Delta"] is None else group["RGB_LiDAR_Delta"],
            "CSI_Detection_ID": value(members["CSI"], "detection_id"),
            "CSI_BBox_X1": bbox_value(members["CSI"], 0),
            "CSI_BBox_Y1": bbox_value(members["CSI"], 1),
            "CSI_BBox_X2": bbox_value(members["CSI"], 2),
            "CSI_BBox_Y2": bbox_value(members["CSI"], 3),
            "CSI_Confidence": value(members["CSI"], "confidence"),
            "CSI_Center_X": value(members["CSI"], "center_x"),
            "CSI_Raw_Bearing": value(members["CSI"], "raw_bearing"),
            "RGB_Detection_ID": value(members["RGB"], "detection_id"),
            "RGB_BBox_X1": bbox_value(members["RGB"], 0),
            "RGB_BBox_Y1": bbox_value(members["RGB"], 1),
            "RGB_BBox_X2": bbox_value(members["RGB"], 2),
            "RGB_BBox_Y2": bbox_value(members["RGB"], 3),
            "RGB_Confidence": value(members["RGB"], "confidence"),
            "RGB_Center_X": value(members["RGB"], "center_x"),
            "RGB_Raw_Bearing": value(members["RGB"], "raw_bearing"),
            "LiDAR_Cluster_ID": value(members["LiDAR"], "cluster_id"),
            "LiDAR_Point_Count": value(members["LiDAR"], "point_count"),
            "LiDAR_Centroid_X": value(members["LiDAR"], "centroid_x"),
            "LiDAR_Centroid_Y": value(members["LiDAR"], "centroid_y"),
            "LiDAR_Min_Distance": value(members["LiDAR"], "min_distance"),
            "LiDAR_Detection_Score": value(
                members["LiDAR"], "LiDAR_detection_score"
            ),
            "Fusion_Time_ms": fusion_time_ms,
            "Frame_Processing_Time_ms": 0.0,
        }
        if set(record) != set(FINAL_OBSTACLE_FIELDS):
            raise AssertionError(
                "Final obstacle record does not match the CSV schema"
            )
        records.append(record)
    return records


def save_camera_features(
    csv_path,
    frame_id,
    weather,
    timestamp,
    layout_id,
    camera_control,
):
    """Append one synchronized pair of precomputed camera quality results."""
    csi_quality = camera_control["csi"]
    rgb_quality = camera_control["rgb"]
    csi_slope = csi_quality["trend_slope"]
    rgb_slope = rgb_quality["trend_slope"]
    lidar_quality = camera_control["lidar"]
    lidar_slope = lidar_quality["trend_slope"]
    row = {
        "Frame_ID": frame_id,
        "Timestamp": f"{timestamp:.4f}",
        "Layout_ID": layout_id,
        "Weather": str(weather),
        "CSI_Sharpness": f"{csi_quality['sharpness']:.4f}",
        "CSI_Contrast": f"{csi_quality['contrast']:.4f}",
        "CSI_Sharpness_Score": f"{csi_quality['sharpness_score']:.6f}",
        "CSI_Contrast_Score": f"{csi_quality['contrast_score']:.6f}",
        "CSI_Quality_Score": f"{csi_quality['quality_score']:.6f}",
        "CSI_Trend": csi_quality["trend"],
        "CSI_Trend_Slope": "" if csi_slope is None else f"{csi_slope:.6f}",
        "CSI_State": csi_quality["state"],
        "CSI_Active": csi_quality["active"],
        "CSI_Weight": f"{csi_quality['weight']:.6f}",
        "RGB_Sharpness": f"{rgb_quality['sharpness']:.4f}",
        "RGB_Contrast": f"{rgb_quality['contrast']:.4f}",
        "RGB_Sharpness_Score": f"{rgb_quality['sharpness_score']:.6f}",
        "RGB_Contrast_Score": f"{rgb_quality['contrast_score']:.6f}",
        "RGB_Quality_Score": f"{rgb_quality['quality_score']:.6f}",
        "RGB_Trend": rgb_quality["trend"],
        "RGB_Trend_Slope": "" if rgb_slope is None else f"{rgb_slope:.6f}",
        "RGB_State": rgb_quality["state"],
        "RGB_Active": rgb_quality["active"],
        "RGB_Weight": f"{rgb_quality['weight']:.6f}",
        "LiDAR_Quality_Score": f"{lidar_quality['quality_score']:.6f}",
        "LiDAR_Trend": lidar_quality["trend"],
        "LiDAR_Trend_Slope": "" if lidar_slope is None else f"{lidar_slope:.6f}",
        "LiDAR_State": lidar_quality["state"],
        "LiDAR_Active": lidar_quality["active"],
        "LiDAR_Weight": f"{lidar_quality['weight']:.6f}",
        "No_Active_Sensor": camera_control["no_active_sensor"],
    }
    with Path(csv_path).open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CAMERA_FEATURE_FIELDS)
        writer.writerow(row)
    print(
        f"Frame: {frame_id:06d}\n"
        f"Weather: {row['Weather']}\n\n"
        f"CSI:\n"
        f"Quality = {row['CSI_Quality_Score']}\n"
        f"Trend = {row['CSI_Trend']}\n"
        f"Slope = {row['CSI_Trend_Slope'] or 'N/A'}\n"
        f"State = {row['CSI_State']}\n"
        f"Active = {row['CSI_Active']}\n"
        f"Weight = {row['CSI_Weight']}\n\n"
        f"RGB:\n"
        f"Quality = {row['RGB_Quality_Score']}\n"
        f"Trend = {row['RGB_Trend']}\n"
        f"Slope = {row['RGB_Trend_Slope'] or 'N/A'}\n"
        f"State = {row['RGB_State']}\n"
        f"Active = {row['RGB_Active']}\n"
        f"Weight = {row['RGB_Weight']}\n\n"
        f"LiDAR:\n"
        f"Quality = {row['LiDAR_Quality_Score']}\n"
        f"Trend = {row['LiDAR_Trend']}\n"
        f"Slope = {row['LiDAR_Trend_Slope'] or 'N/A'}\n"
        f"State = {row['LiDAR_State']}\n"
        f"Active = {row['LiDAR_Active']}\n"
        f"Weight = {row['LiDAR_Weight']}"
    )
    if camera_control["no_active_sensor"]:
        print("Warning: No active sensor available.")


def Save_CSI_Data(image_data, sample_number, output_directory=CSI_DATA_DIRECTORY):
    """Save a Front CSI image using the shared batch number."""
    output_path = Path(output_directory) / (
        f"Front_CSI_Camera_Image_{sample_number:06d}.jpg"
    )
    if not cv2.imwrite(str(output_path), image_data):
        raise OSError(f"Unable to save CSI image: {output_path}")
    return output_path


def Save_RGB_Data(image_data, sample_number, output_directory=RGB_DATA_DIRECTORY):
    """Save an RGB image using the shared batch number."""
    output_path = Path(output_directory) / (
        f"RGB_Camera_Image_{sample_number:06d}.jpg"
    )
    if not cv2.imwrite(str(output_path), image_data):
        raise OSError(f"Unable to save RGB image: {output_path}")
    return output_path


def Lidar_Data(
    angles,
    distances,
    sample_number,
    timestamp=None,
    weather="unknown",
    output_directory=LIDAR_DATA_DIRECTORY,
):
    """Save a Lidar scan using the shared batch number."""
    output_path = Path(output_directory) / (
        f"LiDAR_Data_{sample_number:06d}.npz"
    )
    angles_array = np.asarray(angles)
    distances_array = np.asarray(distances)
    np.savez(
        output_path,
        angles=angles_array,
        distances=distances_array,
        x=np.sin(angles_array) * distances_array,
        y=np.cos(angles_array) * distances_array,
        frame_id=np.int64(sample_number),
        timestamp=np.float64(np.nan if timestamp is None else timestamp),
        weather=np.asarray(str(weather)),
    )
    return output_path


def _sensor_status_lines(sensor_name, sensor_status):
    """Return the three live status lines shared by all sensor displays."""
    weight_label = sensor_status.get("weight_label", "Dynamic Weight")
    return (
        sensor_name,
        f"Quality Score: {float(sensor_status['quality_score']):.3f}",
        f"Trend: {sensor_status['trend']}",
        f"{weight_label}: {float(sensor_status['weight']):.1%}",
    )


def _overlay_sensor_status(image_data, sensor_name, sensor_status):
    """Draw live quality, trend, and weight without modifying the source frame."""
    display_image = np.asarray(image_data).copy()
    if display_image.ndim == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    elif display_image.ndim == 3 and display_image.shape[2] == 1:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
    elif display_image.ndim == 3 and display_image.shape[2] == 4:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2BGR)

    height, width = display_image.shape[:2]
    margin = 12
    line_height = 25
    panel_width = min(
        max(290, int(width * 0.46)), max(1, width - 2 * margin)
    )
    panel_height = min(116, max(1, height - 2 * margin))
    panel = display_image.copy()
    cv2.rectangle(
        panel,
        (margin, margin),
        (margin + panel_width, margin + panel_height),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(panel, 0.68, display_image, 0.32, 0, display_image)

    trend = str(sensor_status["trend"])
    trend_color = STATUS_TREND_COLORS.get(trend, (255, 255, 255))
    for line_number, line in enumerate(
        _sensor_status_lines(sensor_name, sensor_status)
    ):
        color = trend_color if line_number == 2 else (255, 255, 255)
        cv2.putText(
            display_image,
            line,
            (margin + 10, margin + 24 + line_number * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return display_image


def CSI_Camera_Data(image_data, sensor_status=None):
    """Display one Front CSI image with its current adaptive status."""
    display_image = image_data
    if sensor_status is not None:
        display_image = _overlay_sensor_status(
            image_data, "CSI Camera", sensor_status
        )
    cv2.imshow(CSI_FRONT_WINDOW, display_image)


def RGB_Camera_Data(image_data, sensor_status=None):
    """Display one RealSense RGB image with its current adaptive status."""
    display_image = image_data
    if sensor_status is not None:
        display_image = _overlay_sensor_status(
            image_data, "RGB Camera", sensor_status
        )
    cv2.imshow(RGB_WINDOW, display_image)


def close_CSI_Camera_Window():
    """Close the Front CSI OpenCV window if it exists."""
    try:
        cv2.destroyWindow(CSI_FRONT_WINDOW)
    except cv2.error:
        pass


def close_RGB_Camera_Window():
    """Close the RGB OpenCV window if it exists."""
    try:
        cv2.destroyWindow(RGB_WINDOW)
    except cv2.error:
        pass


class LidarDisplay:
    """Display previously acquired QCar2 Lidar scans."""

    def __init__(self, square_size=100):
        self.plot = pg.plot(title="LIDAR")
        self.plot.setXRange(-square_size, square_size)
        self.plot.setYRange(-square_size, square_size)
        self.data = self.plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolBrush="r",
            symbolPen=None,
            symbolSize=4,
        )
        # The ego vehicle is the LiDAR coordinate-system origin, not a return.
        # Show it separately without adding a fabricated point to saved data.
        self.ego_marker = self.plot.plot(
            [0],
            [0],
            pen=None,
            symbol="s",
            symbolBrush=(0, 170, 255),
            symbolPen="w",
            symbolSize=10,
        )
        self.ego_label = pg.TextItem(
            text="QCar1",
            color=(0, 170, 255),
            anchor=(0.5, 1.2),
        )
        self.ego_label.setPos(0, 0)
        self.plot.addItem(self.ego_label)

        self.status_label = pg.TextItem(
            anchor=(0, 0),
            fill=pg.mkBrush(0, 0, 0, 190),
            border=pg.mkPen(220, 220, 220, 180),
        )
        self.status_label.setPos(-0.96 * square_size, 0.96 * square_size)
        self.plot.addItem(self.status_label)
        self._update_status_label(None)

    def _update_status_label(self, sensor_status):
        if sensor_status is None:
            self.status_label.setHtml(
                '<div style="color:white; padding:5px;">'
                "<b>LiDAR</b><br>Awaiting live status...</div>"
            )
            return

        lines = _sensor_status_lines("LiDAR", sensor_status)
        trend = str(sensor_status["trend"])
        rgb_color = STATUS_TREND_COLORS.get(trend, (255, 255, 255))
        trend_color = "#{:02x}{:02x}{:02x}".format(
            rgb_color[2], rgb_color[1], rgb_color[0]
        )
        self.status_label.setHtml(
            '<div style="color:white; padding:5px; font-size:12pt;">'
            f"<b>{lines[0]}</b><br>"
            f"{lines[1]}<br>"
            f'<span style="color:{trend_color};">{lines[2]}</span><br>'
            f"{lines[3]}</div>"
        )

    def update(self, angles, distances, sensor_status=None):
        """Refresh one LiDAR scan and its current adaptive status."""
        x = np.sin(angles) * distances
        y = np.cos(angles) * distances
        self.data.setData(x, y)
        self._update_status_label(sensor_status)

        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.processEvents()

    def process_events(self):
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.processEvents()

    def is_open(self):
        """Return whether the LiDAR preview window is still visible."""
        self.process_events()
        return self.plot.isVisible()

    def close(self):
        self.plot.close()


class SensorBatchRecorder:
    """Acquire and save one synchronized sensor batch every 0.2 seconds.

    QLabs requests are sequential rather than hardware-simultaneous, but all
    sensor results share one batch number. A batch is saved only when LiDAR,
    RGB, and Front CSI acquisitions all succeed.
    """

    def __init__(
        self,
        save_interval=0.2,
        sample_points=400,
        square_size=100,
        scenario_name="unknown_scenario",
        layout_id=None,
        max_frames=None,
    ):
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
        self.camera_controller = CameraAdaptiveController()
        self.experiment_directory = self._create_experiment_directory(
            scenario_name, layout_id
        )
        self.csi_directory = self.experiment_directory / "Front CSI Camera Data"
        self.rgb_directory = self.experiment_directory / "RGB Camera Data"
        self.lidar_directory = self.experiment_directory / "LiDAR Data"
        self.detection_directory = self.experiment_directory / "Detection Results Data"
        self.fusion_directory = self.experiment_directory / "Fusion Results Data"
        self.final_obstacle_directory = (
            self.experiment_directory / "Final Obstacle Results Data"
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
            f"{weather_name}_test{test_number}_{layout_name}_result.csv"
        )
        self.adaptive_fusion_results_path = self.fusion_directory / (
            f"{weather_name}_test{test_number}_{layout_name}_adaptive_fusion.csv"
        )
        self.final_obstacle_results_path = self.final_obstacle_directory / (
            f"{weather_name}_test{test_number}_{layout_name}_final_obstacles.csv"
        )
        self.camera_features_path = (
            self.experiment_directory
            / "Camera Features Data"
            / f"{weather_name}_T{test_number}.csv"
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
        print(f"Saving this run to: {self.experiment_directory}")
        print(f"Camera quality CSV: {self.camera_features_path.name}")
        print(f"Per-sensor detection CSV: {self.detection_results_path.name}")
        print(f"Adaptive fusion CSV: {self.adaptive_fusion_results_path.name}")
        print(f"Final obstacle CSV: {self.final_obstacle_results_path.name}")

    @staticmethod
    def _create_experiment_directory(scenario_name, layout_id=None):
        """Atomically reserve the next test number for one scenario."""
        scenario_name = str(scenario_name).strip()
        if not scenario_name or scenario_name in {".", ".."}:
            raise ValueError("scenario_name must be a non-empty folder name")
        if Path(scenario_name).name != scenario_name:
            raise ValueError("scenario_name cannot contain path separators")

        scenario_directory = EXPERIMENT_DATA_DIRECTORY / scenario_name
        scenario_directory.mkdir(parents=True, exist_ok=True)
        layout_suffix = "" if layout_id is None else f"_layout{int(layout_id)}"
        test_number = 1
        while True:
            # Test numbers are shared across layouts within the same scenario.
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

    def _batch_paths(self, sample_number):
        """Return all expected paths for one shared sample number."""
        return (
            self.csi_directory
            / f"Front_CSI_Camera_Image_{sample_number:06d}.jpg",
            self.rgb_directory / f"RGB_Camera_Image_{sample_number:06d}.jpg",
            self.lidar_directory / f"LiDAR_Data_{sample_number:06d}.npz",
        )

    def _save_complete_batch(
        self,
        angles,
        distances,
        rgb_image,
        csi_image,
        timestamp,
        weather,
        camera_control,
        detection_record,
        fusion_record,
        final_obstacle_records,
    ):
        sample_number = self.next_sample_number
        if int(detection_record["Frame_ID"]) != sample_number:
            raise AssertionError("Detection record Frame_ID does not match sensor batch")
        if int(fusion_record["Frame_ID"]) != sample_number:
            raise AssertionError("Fusion record Frame_ID does not match sensor batch")
        if any(
            int(record["Frame_ID"]) != sample_number
            for record in final_obstacle_records
        ):
            raise AssertionError(
                "Final obstacle record Frame_ID does not match sensor batch"
            )
        try:
            csi_path = Save_CSI_Data(
                csi_image, sample_number, self.csi_directory
            )
            rgb_path = Save_RGB_Data(
                rgb_image, sample_number, self.rgb_directory
            )
            lidar_path = Lidar_Data(
                angles,
                distances,
                sample_number,
                timestamp=timestamp,
                weather=weather,
                output_directory=self.lidar_directory,
            )
            save_camera_features(
                self.camera_features_path,
                sample_number,
                weather,
                timestamp,
                self.layout_id,
                camera_control,
            )
            save_frame_detection_results(
                self.detection_results_path, detection_record
            )
            save_adaptive_fusion_result(
                self.adaptive_fusion_results_path, fusion_record
            )
            save_final_obstacle_results(
                self.final_obstacle_results_path,
                final_obstacle_records,
            )
        except Exception:
            # Roll back any files already written for this incomplete batch.
            for path in self._batch_paths(sample_number):
                path.unlink(missing_ok=True)
            raise

        self.next_sample_number += 1

    def update(self, qcar, weather="unknown"):
        """Acquire a due batch, update displays, and save only if complete.

        Returns False when Escape is pressed in an OpenCV camera window.
        """
        current_time = time.monotonic()
        if (
            self.last_batch_time is not None
            and current_time - self.last_batch_time < self.save_interval
        ):
            self.lidar_display.process_events()
            return (cv2.waitKey(1) & 0xFF) != 27

        # Advance the one shared schedule even if this acquisition fails.
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

        # Calculate quality and controller output for this same acquired frame.
        quality_control_start = time.perf_counter()
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
            # Update histories only for a complete batch that will be saved.
            camera_control = self.camera_controller.evaluate(
                csi_quality, rgb_quality
            )
            quality_control_ms = (
                time.perf_counter() - quality_control_start
            ) * 1000.0

            # Display is deliberately outside Quality_Control_Time_ms.
            self.lidar_display.update(
                angles, distances, camera_control["lidar"]
            )
            RGB_Camera_Data(rgb_image, camera_control["rgb"])
            CSI_Camera_Data(csi_image, camera_control["csi"])

            # Adaptive selection is the detector-execution boundary. Inactive
            # sensors keep an empty result and do not run their detector.
            csi_detections = []
            csi_detection_ms = 0.0
            if camera_control["csi"]["active"]:
                csi_detection_start = time.perf_counter()
                csi_detections = camera_detections_from_image(
                    self.camera_detector, csi_image
                )
                csi_detection_ms = (
                    time.perf_counter() - csi_detection_start
                ) * 1000.0

            rgb_detections = []
            rgb_detection_ms = 0.0
            if camera_control["rgb"]["active"]:
                rgb_detection_start = time.perf_counter()
                rgb_detections = camera_detections_from_image(
                    self.camera_detector, rgb_image
                )
                rgb_detection_ms = (
                    time.perf_counter() - rgb_detection_start
                ) * 1000.0

            lidar_detections = []
            lidar_detection_ms = 0.0
            if camera_control["lidar"]["active"]:
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
                # Finalized immediately after all current-frame saves below.
                "Frame_Processing_Time_ms": 0.0,
                "Quality_Control_Time_ms": quality_control_ms,
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
                camera_control,
                detection_outputs,
                timings,
            )

            # The existing fusion module converts only Active detections to
            # calibrated bearings, associates them, applies binary weighted
            # voting, and accepts candidates supported by at least half of the
            # currently available reliability weight.
            fusion_start = time.perf_counter()
            fusion_result = adaptive_object_fusion(
                frame_id=self.next_sample_number,
                controller_result=camera_control,
                detection_outputs=detection_outputs,
                fusion_threshold=FUSION_THRESHOLD,
            )
            fusion_time_ms = (
                time.perf_counter() - fusion_start
            ) * 1000.0
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
                camera_control,
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
                print(f"Completed collection of {self.max_frames} frames.")
                mean_processing_ms = sum(self.frame_processing_times_ms) / len(
                    self.frame_processing_times_ms
                )
                print(
                    f"Mean Frame_Processing_Time_ms: {mean_processing_ms:.6f}"
                )
                print(
                    f"Mean processing FPS: {1000.0 / mean_processing_ms:.6f}"
                )
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

    def close(self):
        if self.detection_records:
            # The first append for every frame is included in its measured time.
            # Rewrite once on orderly close to persist those finalized values.
            rewrite_frame_detection_results(
                self.detection_results_path, self.detection_records
            )
        if self.adaptive_fusion_records:
            rewrite_adaptive_fusion_results(
                self.adaptive_fusion_results_path,
                self.adaptive_fusion_records,
            )
        if self.final_obstacle_records:
            rewrite_final_obstacle_results(
                self.final_obstacle_results_path,
                self.final_obstacle_records,
            )
        self.lidar_display.close()
        close_CSI_Camera_Window()
        close_RGB_Camera_Window()
