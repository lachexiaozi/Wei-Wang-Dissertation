"""Adaptive object-level late fusion using active-sensor binary voting.

This module consumes, but does not calculate, the controller output from
``SensorQualityController.evaluate`` and the independent detector outputs from
``detection_results.py``.  Adaptive sensor selection is completed before any
detection is converted to a bearing or offered to association.

Detector confidence values are retained in member payloads for diagnostics.
They are deliberately absent from :func:`_binary_weighted_vote`, whose inputs
are only sensor presence bits and controller-provided sensor weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SENSOR_KEYS = ("csi", "rgb", "lidar")
SENSOR_LABELS = {"csi": "CSI", "rgb": "RGB", "lidar": "LiDAR"}
LABEL_TO_KEY = {label: key for key, label in SENSOR_LABELS.items()}
CAMERA_KEYS = ("csi", "rgb")

CSI_WIDTH = 820
CSI_HFOV_DEG = 160.0
CSI_SCALE = 0.7225
CSI_OFFSET_DEG = -0.0987

RGB_WIDTH = 640
RGB_HFOV_DEG = 69.0
RGB_SCALE = 0.9744
RGB_OFFSET_DEG = -1.1331

BEARING_TOLERANCE_DEG = 1.0
WEIGHT_ABS_TOLERANCE = 1e-9

CAMERA_CALIBRATION = {
    "csi": {
        "width": CSI_WIDTH,
        "hfov_deg": CSI_HFOV_DEG,
        "scale": CSI_SCALE,
        "offset_deg": CSI_OFFSET_DEG,
    },
    "rgb": {
        "width": RGB_WIDTH,
        "hfov_deg": RGB_HFOV_DEG,
        "scale": RGB_SCALE,
        "offset_deg": RGB_OFFSET_DEG,
    },
}


@dataclass(frozen=True)
class _DetectionRef:
    """Internal immutable reference used to enforce one-use association."""

    sensor: str
    source_index: int
    bearing_deg: float
    payload: dict[str, Any]

    @property
    def token(self) -> tuple[str, int]:
        return self.sensor, self.source_index


def _finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _validate_fusion_threshold(fusion_threshold: float | None) -> None:
    if fusion_threshold is None:
        return
    threshold = _finite_float(fusion_threshold, "fusion_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("fusion_threshold must be between 0.0 and 1.0")


def _validate_detection_outputs(
    detection_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    missing = set(SENSOR_KEYS) - set(detection_outputs)
    if missing:
        raise KeyError(f"Missing detector outputs: {sorted(missing)}")
    for sensor in SENSOR_KEYS:
        if not isinstance(detection_outputs[sensor], Sequence) or isinstance(
            detection_outputs[sensor], (str, bytes)
        ):
            raise TypeError(f"detection_outputs[{sensor!r}] must be a sequence")


def _adaptive_participants_and_weights(
    controller_result: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, float]]:
    """Read and validate the existing controller result without replacing it."""
    missing = set(SENSOR_KEYS) - set(controller_result)
    if missing:
        raise KeyError(f"Missing controller sensor results: {sorted(missing)}")

    active_keys: list[str] = []
    weights: dict[str, float] = {}
    quality_scores: dict[str, float] = {}
    for sensor in SENSOR_KEYS:
        control = controller_result[sensor]
        for field in ("quality_score", "state", "active", "weight"):
            if field not in control:
                raise KeyError(f"controller_result[{sensor!r}] lacks {field!r}")
        quality_scores[sensor] = _finite_float(
            control["quality_score"], f"{sensor}.quality_score"
        )
        weight = _finite_float(control["weight"], f"{sensor}.weight")
        if weight < 0.0:
            raise ValueError(f"{sensor}.weight cannot be negative")
        weights[sensor] = weight

        if bool(control["active"]):
            active_keys.append(sensor)
        elif not math.isclose(
            weight, 0.0, rel_tol=0.0, abs_tol=WEIGHT_ABS_TOLERANCE
        ):
            raise AssertionError("Inactive sensor must have weight == 0.0")

    # Selection has already happened: inactive keys are absent by construction.
    for sensor in SENSOR_KEYS:
        if not bool(controller_result[sensor]["active"]):
            assert sensor not in active_keys

    if active_keys:
        active_weight_sum = sum(weights[sensor] for sensor in active_keys)
        if not math.isclose(
            active_weight_sum, 1.0, rel_tol=0.0, abs_tol=WEIGHT_ABS_TOLERANCE
        ):
            raise AssertionError("Weights of active sensors must sum to 1.0")

        squared_total = sum(quality_scores[sensor] ** 2 for sensor in active_keys)
        if squared_total <= 0.0:
            raise AssertionError("Active q^2 total must be positive")
        for sensor in active_keys:
            expected = quality_scores[sensor] ** 2 / squared_total
            if not math.isclose(
                weights[sensor],
                expected,
                rel_tol=1e-9,
                abs_tol=WEIGHT_ABS_TOLERANCE,
            ):
                raise AssertionError(
                    "Controller weight does not equal active-only q^2 normalization"
                )
    else:
        if any(
            not math.isclose(
                weights[sensor],
                0.0,
                rel_tol=0.0,
                abs_tol=WEIGHT_ABS_TOLERANCE,
            )
            for sensor in SENSOR_KEYS
        ):
            raise AssertionError("All weights must be zero with no active sensor")

    if "no_active_sensor" in controller_result:
        if bool(controller_result["no_active_sensor"]) != (not active_keys):
            raise AssertionError("no_active_sensor disagrees with active flags")
    return active_keys, weights


def _prepare_camera_detection(
    sensor: str,
    source_index: int,
    detection: Mapping[str, Any],
) -> _DetectionRef:
    if not isinstance(detection, Mapping):
        raise TypeError(f"{sensor} detection must be a mapping")
    try:
        bbox = [_finite_float(value, f"{sensor}.bbox") for value in detection["bbox"]]
    except KeyError as exc:
        raise KeyError(f"{sensor} detection lacks 'bbox'") from exc
    if len(bbox) != 4:
        raise ValueError(f"{sensor} bbox must contain four coordinates")

    calibration = CAMERA_CALIBRATION[sensor]
    center_x = (bbox[0] + bbox[2]) / 2.0
    raw_bearing = (
        (center_x - calibration["width"] / 2.0)
        / (calibration["width"] / 2.0)
        * (calibration["hfov_deg"] / 2.0)
    )
    calibrated_bearing = (
        calibration["scale"] * raw_bearing + calibration["offset_deg"]
    )

    payload = dict(detection)
    payload["bbox"] = bbox
    payload.update(
        {
            "detection_id": source_index + 1,
            "center_x": center_x,
            "raw_bearing": raw_bearing,
            "calibrated_bearing": calibrated_bearing,
        }
    )
    return _DetectionRef(sensor, source_index, calibrated_bearing, payload)


def _prepare_lidar_detection(
    source_index: int,
    detection: Mapping[str, Any],
) -> _DetectionRef:
    if not isinstance(detection, Mapping):
        raise TypeError("lidar detection must be a mapping")
    try:
        centroid_x = _finite_float(detection["centroid_x"], "lidar.centroid_x")
        centroid_y = _finite_float(detection["centroid_y"], "lidar.centroid_y")
    except KeyError as exc:
        raise KeyError(f"LiDAR detection lacks {exc.args[0]!r}") from exc
    bearing = math.degrees(math.atan2(centroid_x, centroid_y))

    payload = dict(detection)
    payload.update(
        {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bearing_deg": bearing,
        }
    )
    return _DetectionRef("lidar", source_index, bearing, payload)


def _prepare_participating_detections(
    participant_keys: Sequence[str],
    detection_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[_DetectionRef]]:
    """Convert only participant detections; excluded sensors are never read here."""
    prepared = {sensor: [] for sensor in SENSOR_KEYS}
    for sensor in participant_keys:
        for source_index, detection in enumerate(detection_outputs[sensor]):
            if sensor in CAMERA_KEYS:
                reference = _prepare_camera_detection(
                    sensor, source_index, detection
                )
            else:
                reference = _prepare_lidar_detection(source_index, detection)
            prepared[sensor].append(reference)
    return prepared


def _empty_group() -> dict[str, Any]:
    return {
        "members": {sensor: None for sensor in SENSOR_KEYS},
        "association_edges": [],
    }


def _single_member_group(reference: _DetectionRef) -> dict[str, Any]:
    group = _empty_group()
    group["members"][reference.sensor] = reference
    return group


def _attach_camera_to_lidar_groups(
    camera_references: Sequence[_DetectionRef],
    groups: list[dict[str, Any]],
) -> list[_DetectionRef]:
    """Globally greedy one-to-one matching for one camera against LiDAR groups."""
    if not camera_references or not groups:
        return list(camera_references)
    sensor = camera_references[0].sensor
    candidates: list[tuple[float, int, int]] = []
    for camera_index, camera in enumerate(camera_references):
        for group_index, group in enumerate(groups):
            lidar = group["members"]["lidar"]
            if lidar is None:
                continue
            delta = abs(camera.bearing_deg - lidar.bearing_deg)
            if delta <= BEARING_TOLERANCE_DEG:
                candidates.append((delta, camera_index, group_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_cameras: set[int] = set()
    used_groups: set[int] = set()
    for delta, camera_index, group_index in candidates:
        if camera_index in used_cameras or group_index in used_groups:
            continue
        camera = camera_references[camera_index]
        group = groups[group_index]
        assert group["members"][sensor] is None
        group["members"][sensor] = camera
        group["association_edges"].append(
            {
                "sensors": [SENSOR_LABELS[sensor], "LiDAR"],
                "bearing_difference_deg": delta,
            }
        )
        used_cameras.add(camera_index)
        used_groups.add(group_index)

    return [
        camera
        for index, camera in enumerate(camera_references)
        if index not in used_cameras
    ]


def _match_unmatched_cameras(
    csi_references: Sequence[_DetectionRef],
    rgb_references: Sequence[_DetectionRef],
) -> tuple[list[dict[str, Any]], list[_DetectionRef], list[_DetectionRef]]:
    """Globally greedy nearest-bearing CSI/RGB one-to-one matching."""
    candidates: list[tuple[float, int, int]] = []
    for csi_index, csi in enumerate(csi_references):
        for rgb_index, rgb in enumerate(rgb_references):
            delta = abs(csi.bearing_deg - rgb.bearing_deg)
            if delta <= BEARING_TOLERANCE_DEG:
                candidates.append((delta, csi_index, rgb_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_csi: set[int] = set()
    used_rgb: set[int] = set()
    matched_groups: list[dict[str, Any]] = []
    for delta, csi_index, rgb_index in candidates:
        if csi_index in used_csi or rgb_index in used_rgb:
            continue
        group = _empty_group()
        group["members"]["csi"] = csi_references[csi_index]
        group["members"]["rgb"] = rgb_references[rgb_index]
        group["association_edges"].append(
            {
                "sensors": ["CSI", "RGB"],
                "bearing_difference_deg": delta,
            }
        )
        matched_groups.append(group)
        used_csi.add(csi_index)
        used_rgb.add(rgb_index)

    unmatched_csi = [
        detection
        for index, detection in enumerate(csi_references)
        if index not in used_csi
    ]
    unmatched_rgb = [
        detection
        for index, detection in enumerate(rgb_references)
        if index not in used_rgb
    ]
    return matched_groups, unmatched_csi, unmatched_rgb


def _associate_active_detections(
    participant_keys: Sequence[str],
    prepared: Mapping[str, list[_DetectionRef]],
) -> list[dict[str, Any]]:
    """Apply LiDAR-anchor grouping, then camera-camera matching, then singles."""
    groups: list[dict[str, Any]] = []
    unmatched_csi = list(prepared["csi"])
    unmatched_rgb = list(prepared["rgb"])

    if "lidar" in participant_keys:
        groups = [
            _single_member_group(reference) for reference in prepared["lidar"]
        ]
        if "csi" in participant_keys:
            unmatched_csi = _attach_camera_to_lidar_groups(
                unmatched_csi, groups
            )
        if "rgb" in participant_keys:
            unmatched_rgb = _attach_camera_to_lidar_groups(
                unmatched_rgb, groups
            )

    if "csi" in participant_keys and "rgb" in participant_keys:
        camera_groups, unmatched_csi, unmatched_rgb = _match_unmatched_cameras(
            unmatched_csi, unmatched_rgb
        )
        groups.extend(camera_groups)

    if "csi" in participant_keys:
        groups.extend(_single_member_group(item) for item in unmatched_csi)
    if "rgb" in participant_keys:
        groups.extend(_single_member_group(item) for item in unmatched_rgb)
    return groups


def _binary_weighted_vote(
    sensor_presence: Mapping[str, int],
    participant_keys: Sequence[str],
    weights: Mapping[str, float],
) -> float:
    """Return sum(w_i * d_i) over participants; no detector score is accepted."""
    return float(
        sum(weights[sensor] * sensor_presence[sensor] for sensor in participant_keys)
    )


def _pair_delta(
    members: Mapping[str, _DetectionRef | None], first: str, second: str
) -> float | None:
    if members[first] is None or members[second] is None:
        return None
    return abs(members[first].bearing_deg - members[second].bearing_deg)


def _serialize_groups(
    groups: Sequence[dict[str, Any]],
    participant_keys: Sequence[str],
    weights: Mapping[str, float],
    fusion_threshold: float | None,
) -> list[dict[str, Any]]:
    participant_set = set(participant_keys)
    active_labels = [SENSOR_LABELS[sensor] for sensor in participant_keys]
    serialized: list[dict[str, Any]] = []

    for group_id, group in enumerate(groups, start=1):
        members: dict[str, _DetectionRef | None] = group["members"]
        presence = {
            sensor: int(members[sensor] is not None) for sensor in SENSOR_KEYS
        }
        fusion_score = _binary_weighted_vote(
            presence, participant_keys, weights
        )
        bearings = [
            member.bearing_deg for member in members.values() if member is not None
        ]
        public_members = {
            SENSOR_LABELS[sensor]: (
                None if members[sensor] is None else dict(members[sensor].payload)
            )
            for sensor in SENSOR_KEYS
        }
        item = {
            "Group_ID": group_id,
            "members": public_members,
            "sensor_presence": {
                SENSOR_LABELS[sensor]: presence[sensor] for sensor in SENSOR_KEYS
            },
            "active_sensors": list(active_labels),
            "Representative_Bearing_deg": sum(bearings) / len(bearings),
            "CSI_Participating": "csi" in participant_set,
            "RGB_Participating": "rgb" in participant_set,
            "LiDAR_Participating": "lidar" in participant_set,
            "CSI_Detected": bool(presence["csi"]),
            "RGB_Detected": bool(presence["rgb"]),
            "LiDAR_Detected": bool(presence["lidar"]),
            "CSI_Calibrated_Bearing": (
                None if members["csi"] is None else members["csi"].bearing_deg
            ),
            "RGB_Calibrated_Bearing": (
                None if members["rgb"] is None else members["rgb"].bearing_deg
            ),
            "LiDAR_Bearing": (
                None if members["lidar"] is None else members["lidar"].bearing_deg
            ),
            "CSI_RGB_Delta": _pair_delta(members, "csi", "rgb"),
            "CSI_LiDAR_Delta": _pair_delta(members, "csi", "lidar"),
            "RGB_LiDAR_Delta": _pair_delta(members, "rgb", "lidar"),
            "CSI_Weight": weights["csi"],
            "RGB_Weight": weights["rgb"],
            "LiDAR_Weight": weights["lidar"],
            "Fusion_Score": fusion_score,
            "Fusion_Decision": (
                None
                if fusion_threshold is None
                else fusion_score >= float(fusion_threshold)
            ),
            "association_edges": [dict(edge) for edge in group["association_edges"]],
        }
        serialized.append(item)
    return serialized


def _validate_result_groups(
    groups: Sequence[dict[str, Any]],
    participant_keys: Sequence[str],
    weights: Mapping[str, float],
    prepared: Mapping[str, Sequence[_DetectionRef]],
) -> None:
    participant_set = set(participant_keys)
    seen_tokens: set[tuple[str, int]] = set()
    for group in groups:
        members: Mapping[str, _DetectionRef | None] = group["members"]
        if set(members) != set(SENSOR_KEYS):
            raise AssertionError("Every group must have exactly one slot per sensor")
        for sensor, member in members.items():
            if sensor not in participant_set and member is not None:
                raise AssertionError(
                    "Inactive sensor detection entered an adaptive object group"
                )
            if member is not None:
                if member.sensor != sensor:
                    raise AssertionError("Detection stored in the wrong sensor slot")
                if member.token in seen_tokens:
                    raise AssertionError("A detection belongs to more than one group")
                seen_tokens.add(member.token)

        for edge in group["association_edges"]:
            difference = float(edge["bearing_difference_deg"])
            if difference > BEARING_TOLERANCE_DEG + 1e-12:
                raise AssertionError("Matched pair exceeds 1.0 degree gate")

        presence = {
            sensor: int(members[sensor] is not None) for sensor in SENSOR_KEYS
        }
        score = _binary_weighted_vote(presence, participant_keys, weights)
        expected = sum(
            weights[sensor] for sensor in participant_keys if presence[sensor]
        )
        if not math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("Fusion score is not a binary weighted vote")

    expected_tokens = {
        reference.token
        for sensor in participant_keys
        for reference in prepared[sensor]
    }
    if seen_tokens != expected_tokens:
        raise AssertionError(
            "Every participating detection must appear in exactly one object group"
        )


def _run_object_fusion(
    participant_keys: Sequence[str],
    weights: Mapping[str, float],
    detection_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    fusion_threshold: float | None,
) -> list[dict[str, Any]]:
    """Shared geometry only; callers explicitly choose adaptive or fixed inputs."""
    _validate_detection_outputs(detection_outputs)
    _validate_fusion_threshold(fusion_threshold)
    if not participant_keys:
        return []

    # This is the selection boundary. Only participant keys cross it.
    prepared = _prepare_participating_detections(
        participant_keys, detection_outputs
    )
    groups = _associate_active_detections(participant_keys, prepared)
    _validate_result_groups(groups, participant_keys, weights, prepared)
    return _serialize_groups(groups, participant_keys, weights, fusion_threshold)


def adaptive_object_fusion(
    frame_id: int,
    controller_result: Mapping[str, Mapping[str, Any]],
    detection_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    fusion_threshold: float | None = None,
) -> dict[str, Any]:
    """Fuse one frame after applying the existing controller's selection.

    ``controller_result`` must be the unmodified output of
    ``SensorQualityController.evaluate``.  Its state/activity/weights are read
    directly and validated; this module does not classify reliability or
    calculate a replacement set of weights.
    """
    participant_keys, weights = _adaptive_participants_and_weights(
        controller_result
    )
    groups = _run_object_fusion(
        participant_keys, weights, detection_outputs, fusion_threshold
    )
    active_labels = [SENSOR_LABELS[sensor] for sensor in participant_keys]

    result: dict[str, Any] = {
        "Branch": "Adaptive",
        "Frame_ID": int(frame_id),
        "CSI_q": float(controller_result["csi"]["quality_score"]),
        "RGB_q": float(controller_result["rgb"]["quality_score"]),
        "LiDAR_q": float(controller_result["lidar"]["quality_score"]),
        "CSI_State": controller_result["csi"]["state"],
        "RGB_State": controller_result["rgb"]["state"],
        "LiDAR_State": controller_result["lidar"]["state"],
        "CSI_Active": bool(controller_result["csi"]["active"]),
        "RGB_Active": bool(controller_result["rgb"]["active"]),
        "LiDAR_Active": bool(controller_result["lidar"]["active"]),
        "CSI_Weight": weights["csi"],
        "RGB_Weight": weights["rgb"],
        "LiDAR_Weight": weights["lidar"],
        "Active_Sensors": active_labels,
        "Number_of_Object_Groups": len(groups),
        "No_Active_Sensor": not participant_keys,
        "Bearing_Tolerance_deg": BEARING_TOLERANCE_DEG,
        "Fusion_Threshold": fusion_threshold,
        "Object_Groups": groups,
        "Object_Groups_JSON": json.dumps(
            groups, separators=(",", ":"), ensure_ascii=False
        ),
    }
    return result


def fixed_weight_fusion(
    frame_id: int,
    detection_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    fusion_threshold: float | None = None,
) -> dict[str, Any]:
    """Independent fixed baseline: all sensors participate at weight 1/3.

    This function does not accept or consult adaptive controller output.  It
    shares only bearing preparation, the 1-degree association, and binary vote
    mechanics with the adaptive branch.
    """
    participant_keys = list(SENSOR_KEYS)
    weights = {sensor: 1.0 / 3.0 for sensor in SENSOR_KEYS}
    groups = _run_object_fusion(
        participant_keys, weights, detection_outputs, fusion_threshold
    )
    return {
        "Branch": "Fixed",
        "Frame_ID": int(frame_id),
        "CSI_Weight": weights["csi"],
        "RGB_Weight": weights["rgb"],
        "LiDAR_Weight": weights["lidar"],
        "Active_Sensors": [SENSOR_LABELS[sensor] for sensor in SENSOR_KEYS],
        "Number_of_Object_Groups": len(groups),
        "No_Active_Sensor": False,
        "Bearing_Tolerance_deg": BEARING_TOLERANCE_DEG,
        "Fusion_Threshold": fusion_threshold,
        "Object_Groups": groups,
        "Object_Groups_JSON": json.dumps(
            groups, separators=(",", ":"), ensure_ascii=False
        ),
    }
