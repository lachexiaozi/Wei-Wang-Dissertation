"""Simple scan-order LiDAR obstacle clustering for QLabs data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_FORWARD_Y_M = 1.0
MAX_FORWARD_Y_M = 40.0
MAX_ABS_X_M = 7.0
MAX_INDEX_GAP = 2
CLUSTER_DISTANCE_M = 1.0


@dataclass(frozen=True)
class LidarPoint:
    scan_index: int
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class ObstacleCluster:
    cluster_id: int
    point_count: int
    centroid_x: float
    centroid_y: float
    min_distance: float
    LiDAR_detection_score: float
    class_name: str = "Obstacle"


@dataclass(frozen=True)
class DetectionResult:
    roi_points: tuple[LidarPoint, ...]
    clusters: tuple[ObstacleCluster, ...]
    point_cluster_ids: tuple[int, ...]


def detection_score(point_count: int) -> float:
    """Return the fixed heuristic detection score (not a probability)."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if point_count == 1:
        return 0.4
    if point_count == 2:
        return 0.7
    return 0.9


def filter_front_roi(angles: np.ndarray, distances: np.ndarray) -> list[LidarPoint]:
    """Convert a scan to x/y and retain valid points in the fixed front ROI."""
    angle_values = np.asarray(angles, dtype=float).reshape(-1)
    distance_values = np.asarray(distances, dtype=float).reshape(-1)
    if angle_values.shape != distance_values.shape:
        raise ValueError(
            "angles and distances must contain the same number of elements: "
            f"{angle_values.shape} != {distance_values.shape}"
        )

    finite = np.isfinite(angle_values) & np.isfinite(distance_values)
    valid_distance = distance_values > 0
    x_values = distance_values * np.sin(angle_values)
    y_values = distance_values * np.cos(angle_values)
    in_roi = (
        (y_values >= MIN_FORWARD_Y_M)
        & (y_values <= MAX_FORWARD_Y_M)
        & (np.abs(x_values) <= MAX_ABS_X_M)
    )
    valid_indices = np.flatnonzero(finite & valid_distance & in_roi)

    return [
        LidarPoint(
            scan_index=int(index),
            x=float(x_values[index]),
            y=float(y_values[index]),
            distance=float(distance_values[index]),
        )
        for index in valid_indices
    ]


def _summarize_cluster(
    cluster_id: int, points: list[LidarPoint]
) -> ObstacleCluster:
    return ObstacleCluster(
        cluster_id=cluster_id,
        point_count=len(points),
        centroid_x=float(np.mean([point.x for point in points])),
        centroid_y=float(np.mean([point.y for point in points])),
        min_distance=min(point.distance for point in points),
        LiDAR_detection_score=detection_score(len(points)),
    )


def _merge_circular_boundary_clusters(
    point_groups: list[list[LidarPoint]], scan_length: int
) -> list[list[LidarPoint]]:
    """Merge only the first/last clusters when they cross the scan boundary."""
    if len(point_groups) < 2:
        return point_groups

    first_boundary_point = min(point_groups[0], key=lambda point: point.scan_index)
    last_boundary_point = max(point_groups[-1], key=lambda point: point.scan_index)
    circular_index_gap = (
        first_boundary_point.scan_index + scan_length - last_boundary_point.scan_index
    )
    boundary_distance = float(
        np.hypot(
            first_boundary_point.x - last_boundary_point.x,
            first_boundary_point.y - last_boundary_point.y,
        )
    )

    if (
        circular_index_gap <= MAX_INDEX_GAP
        and boundary_distance <= CLUSTER_DISTANCE_M
    ):
        merged_boundary_group = point_groups[0] + point_groups[-1]
        return [merged_boundary_group, *point_groups[1:-1]]
    return point_groups


def detect_obstacles(angles: np.ndarray, distances: np.ndarray) -> DetectionResult:
    """Cluster fixed-ROI points using only scan gap and Euclidean distance."""
    roi_points = filter_front_roi(angles, distances)
    if not roi_points:
        return DetectionResult(roi_points=(), clusters=(), point_cluster_ids=())

    point_groups: list[list[LidarPoint]] = [[roi_points[0]]]
    for point in roi_points[1:]:
        previous = point_groups[-1][-1]
        index_gap = point.scan_index - previous.scan_index
        euclidean_distance = float(np.hypot(point.x - previous.x, point.y - previous.y))
        if index_gap <= MAX_INDEX_GAP and euclidean_distance <= CLUSTER_DISTANCE_M:
            point_groups[-1].append(point)
        else:
            point_groups.append([point])

    point_groups = _merge_circular_boundary_clusters(
        point_groups, np.asarray(angles).size
    )
    clusters = tuple(
        _summarize_cluster(cluster_id, points)
        for cluster_id, points in enumerate(point_groups, start=1)
    )
    cluster_id_by_scan_index = {
        point.scan_index: cluster_id
        for cluster_id, points in enumerate(point_groups, start=1)
        for point in points
    }
    point_cluster_ids = tuple(
        cluster_id_by_scan_index[point.scan_index] for point in roi_points
    )
    return DetectionResult(
        roi_points=tuple(roi_points),
        clusters=clusters,
        point_cluster_ids=point_cluster_ids,
    )
