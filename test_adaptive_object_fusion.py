"""Artificial, detector-free checks for adaptive object-level fusion."""

from __future__ import annotations

import copy
import math
import unittest

from adaptive_object_fusion import (
    CAMERA_CALIBRATION,
    adaptive_object_fusion,
    fixed_weight_fusion,
)
from camera_adaptive_control import SensorQualityController


def camera_detection(sensor: str, calibrated_bearing: float, confidence=0.5):
    calibration = CAMERA_CALIBRATION[sensor]
    raw_bearing = (
        calibrated_bearing - calibration["offset_deg"]
    ) / calibration["scale"]
    center_x = (
        calibration["width"] / 2.0
        + raw_bearing
        / (calibration["hfov_deg"] / 2.0)
        * (calibration["width"] / 2.0)
    )
    return {
        "bbox": [center_x - 10.0, 100.0, center_x + 10.0, 140.0],
        "confidence": confidence,
    }


def lidar_detection(cluster_id: int, bearing: float, detection_score=0.9):
    forward = 10.0
    return {
        "cluster_id": cluster_id,
        "point_count": 3,
        "centroid_x": math.tan(math.radians(bearing)) * forward,
        "centroid_y": forward,
        "min_distance": forward,
        "LiDAR_detection_score": detection_score,
    }


def detector_outputs(csi=(), rgb=(), lidar=()):
    return {"csi": list(csi), "rgb": list(rgb), "lidar": list(lidar)}


def controller_result(csi_q: float, rgb_q: float):
    return SensorQualityController().evaluate(
        {"quality_score": csi_q}, {"quality_score": rgb_q}
    )


class AdaptiveObjectFusionTests(unittest.TestCase):
    def test_camera_raw_and_calibrated_bearings_are_retained(self):
        result = adaptive_object_fusion(
            0,
            controller_result(0.95, 0.92),
            detector_outputs(
                [camera_detection("csi", 12.0)],
                [camera_detection("rgb", -8.0)],
                [],
            ),
        )
        csi_member = result["Object_Groups"][0]["members"]["CSI"]
        rgb_member = result["Object_Groups"][1]["members"]["RGB"]
        self.assertIn("raw_bearing", csi_member)
        self.assertIn("calibrated_bearing", csi_member)
        self.assertAlmostEqual(csi_member["calibrated_bearing"], 12.0)
        self.assertAlmostEqual(rgb_member["calibrated_bearing"], -8.0)

    def test_three_active_sensors_form_one_group(self):
        result = adaptive_object_fusion(
            1,
            controller_result(0.95, 0.92),
            detector_outputs(
                [camera_detection("csi", 0.2)],
                [camera_detection("rgb", -0.1)],
                [lidar_detection(1, 0.0)],
            ),
        )
        self.assertEqual(result["Active_Sensors"], ["CSI", "RGB", "LiDAR"])
        self.assertEqual(result["Number_of_Object_Groups"], 1)
        group = result["Object_Groups"][0]
        self.assertTrue(group["CSI_Detected"])
        self.assertTrue(group["RGB_Detected"])
        self.assertTrue(group["LiDAR_Detected"])
        self.assertAlmostEqual(group["Fusion_Score"], 1.0)

    def test_two_active_sensors_exclude_inactive_camera(self):
        controls = controller_result(0.68, 0.86)
        outputs = detector_outputs(
            # Deliberately malformed: selection must exclude it before bearing.
            [{"confidence": 0.999}],
            [camera_detection("rgb", 0.2)],
            [lidar_detection(1, 0.0)],
        )
        result = adaptive_object_fusion(2, controls, outputs)
        self.assertEqual(result["Active_Sensors"], ["RGB", "LiDAR"])
        self.assertEqual(result["CSI_Weight"], 0.0)
        self.assertEqual(result["Number_of_Object_Groups"], 1)
        group = result["Object_Groups"][0]
        self.assertFalse(group["CSI_Participating"])
        self.assertIsNone(group["members"]["CSI"])
        self.assertAlmostEqual(group["Fusion_Score"], 1.0)

    def test_one_active_sensor_has_unit_weight_and_unit_score(self):
        result = adaptive_object_fusion(
            3,
            controller_result(0.60, 0.70),
            detector_outputs(
                [camera_detection("csi", 0.0)],
                [camera_detection("rgb", 0.0)],
                [lidar_detection(1, 4.0)],
            ),
        )
        self.assertEqual(result["Active_Sensors"], ["LiDAR"])
        self.assertAlmostEqual(result["LiDAR_Weight"], 1.0)
        self.assertEqual(result["Number_of_Object_Groups"], 1)
        self.assertAlmostEqual(result["Object_Groups"][0]["Fusion_Score"], 1.0)

    def test_unmatched_lidar_is_retained_as_single_sensor_group(self):
        result = adaptive_object_fusion(
            4,
            controller_result(0.95, 0.92),
            detector_outputs(
                [camera_detection("csi", 0.1)],
                [camera_detection("rgb", -0.1)],
                [lidar_detection(1, 0.0), lidar_detection(2, 10.0)],
            ),
        )
        self.assertEqual(result["Number_of_Object_Groups"], 2)
        lidar_only = result["Object_Groups"][1]
        self.assertFalse(lidar_only["CSI_Detected"])
        self.assertFalse(lidar_only["RGB_Detected"])
        self.assertTrue(lidar_only["LiDAR_Detected"])
        self.assertAlmostEqual(
            lidar_only["Fusion_Score"], result["LiDAR_Weight"]
        )

    def test_no_active_sensor_is_safe(self):
        controls = {
            "csi": {
                "quality_score": 0.1,
                "state": "Unreliable",
                "active": False,
                "weight": 0.0,
            },
            "rgb": {
                "quality_score": 0.2,
                "state": "Unreliable",
                "active": False,
                "weight": 0.0,
            },
            "lidar": {
                "quality_score": 0.0,
                "state": "Unreliable",
                "active": False,
                "weight": 0.0,
            },
            "no_active_sensor": True,
        }
        result = adaptive_object_fusion(
            5,
            controls,
            detector_outputs(
                [{"malformed": True}],
                [{"malformed": True}],
                [{"malformed": True}],
            ),
        )
        self.assertTrue(result["No_Active_Sensor"])
        self.assertEqual(result["Active_Sensors"], [])
        self.assertEqual(result["Object_Groups"], [])

    def test_detector_scores_cannot_change_binary_fusion_score(self):
        controls = controller_result(0.95, 0.92)
        outputs = detector_outputs(
            [camera_detection("csi", 0.2, confidence=0.01)],
            [camera_detection("rgb", -0.1, confidence=0.02)],
            [lidar_detection(1, 0.0, detection_score=0.4)],
        )
        changed = copy.deepcopy(outputs)
        changed["csi"][0]["confidence"] = 0.99
        changed["rgb"][0]["confidence"] = 0.98
        changed["lidar"][0]["LiDAR_detection_score"] = 0.9
        first = adaptive_object_fusion(6, controls, outputs)
        second = adaptive_object_fusion(6, controls, changed)
        self.assertEqual(
            first["Object_Groups"][0]["Fusion_Score"],
            second["Object_Groups"][0]["Fusion_Score"],
        )

    def test_fixed_baseline_is_all_sensor_equal_weight_branch(self):
        outputs = detector_outputs(
            [camera_detection("csi", 0.2)],
            [camera_detection("rgb", -0.1)],
            [lidar_detection(1, 0.0)],
        )
        result = fixed_weight_fusion(7, outputs)
        self.assertEqual(result["Branch"], "Fixed")
        self.assertEqual(result["Active_Sensors"], ["CSI", "RGB", "LiDAR"])
        self.assertAlmostEqual(result["CSI_Weight"], 1.0 / 3.0)
        self.assertAlmostEqual(result["Object_Groups"][0]["Fusion_Score"], 1.0)

    def test_threshold_is_optional(self):
        result = adaptive_object_fusion(
            8,
            controller_result(0.95, 0.92),
            detector_outputs(lidar=[lidar_detection(1, 0.0)]),
        )
        self.assertIsNone(result["Object_Groups"][0]["Fusion_Decision"])

    def test_association_gate_is_exactly_one_degree(self):
        controls = controller_result(0.95, 0.92)
        at_gate = adaptive_object_fusion(
            9,
            controls,
            detector_outputs(
                [camera_detection("csi", 0.0)],
                [camera_detection("rgb", 1.0)],
                [],
            ),
        )
        outside_gate = adaptive_object_fusion(
            10,
            controls,
            detector_outputs(
                [camera_detection("csi", 0.0)],
                [camera_detection("rgb", 1.0001)],
                [],
            ),
        )
        self.assertEqual(at_gate["Number_of_Object_Groups"], 1)
        self.assertEqual(outside_gate["Number_of_Object_Groups"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
