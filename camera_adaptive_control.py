"""Camera quality state and adaptive weighting for the QCar2 recorder.

Weather is deliberately absent from this module: it is an experiment annotation
and must not influence any real-time quality or control decision.
"""

from collections import deque

import numpy as np


# Camera quality/controller configuration (edit these values for calibration).
HIGH_THRESHOLD = 0.90
LOW_THRESHOLD = 0.75

TREND_WINDOW = 5
TREND_DELTA = 0.001
LIDAR_QUALITY = 1.0

CAMERA_REFERENCES = {
    1: {
        "RGB": {"sharpness": 314.6, "contrast": 29.4},
        "CSI": {"sharpness": 77.1, "contrast": 25.2},
    },
    2: {
        "RGB": {"sharpness": 313.1, "contrast": 30.3},
        "CSI": {"sharpness": 76.0, "contrast": 25.9},
    },
    3: {
        "RGB": {"sharpness": 317.9, "contrast": 28.9},
        "CSI": {"sharpness": 75.4, "contrast": 25.5},
    },
}


def get_camera_references(layout_id):
    """Return the fixed clear-weather references for an explicit layout."""
    try:
        return CAMERA_REFERENCES[int(layout_id)]
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("layout_id must be explicitly set to 1, 2, or 3") from exc


def calculate_trend(quality_history, trend_delta=TREND_DELTA):
    """Classify the least-squares slope over the latest five quality scores."""
    scores = list(quality_history)
    if len(scores) < TREND_WINDOW:
        return {"trend": "Initializing", "slope": None}

    scores = np.asarray(scores[-TREND_WINDOW:], dtype=float)
    slope = float(
        np.polyfit(np.arange(TREND_WINDOW, dtype=float), scores, 1)[0]
    )
    if slope > trend_delta:
        trend = "Improving"
    elif slope < -trend_delta:
        trend = "Degrading"
    else:
        trend = "Stable"
    return {"trend": trend, "slope": slope}


def classify_sensor_state(
    quality_score,
    trend,
    high_threshold=HIGH_THRESHOLD,
    low_threshold=LOW_THRESHOLD,
):
    """Determine state and activity from current quality only.

    ``trend`` is accepted to keep the control result self-describing, but V1
    intentionally does not use it to change state, activity, or weight.
    """
    del trend
    if quality_score >= high_threshold:
        state = "Reliable"
    elif quality_score >= low_threshold:
        state = "Degraded"
    else:
        state = "Unreliable"
    return {"state": state, "active": quality_score >= low_threshold}


# Backwards-compatible name for existing callers.
determine_sensor_state = classify_sensor_state


def calculate_sensor_weights(sensor_results):
    """Normalize squared quality across every active sensor in one operation."""
    active_names = [
        name for name, result in sensor_results.items() if result["active"]
    ]
    for result in sensor_results.values():
        result["weight"] = 0.0

    if not active_names:
        return True
    if len(active_names) == 1:
        sensor_results[active_names[0]]["weight"] = 1.0
        return False

    squared_scores = {
        name: sensor_results[name]["quality_score"] ** 2
        for name in active_names
    }
    total = sum(squared_scores.values())
    if total <= 0:
        return True
    for name in active_names:
        sensor_results[name]["weight"] = squared_scores[name] / total
    return False


class SensorQualityController:
    """Evaluate synchronized CSI, RGB, and simulation-specific LiDAR quality."""

    def __init__(
        self,
        high_threshold=HIGH_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
        trend_delta=TREND_DELTA,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.trend_delta = trend_delta
        self.quality_histories = {
            "csi": deque(maxlen=TREND_WINDOW),
            "rgb": deque(maxlen=TREND_WINDOW),
            "lidar": deque(maxlen=TREND_WINDOW),
        }

    def evaluate(self, csi_quality, rgb_quality):
        """Calculate trend, state, activity, and unified three-sensor weights."""
        inputs = {
            "csi": csi_quality,
            "rgb": rgb_quality,
            "lidar": {"quality_score": LIDAR_QUALITY},
        }
        sensor_results = {}

        for name, quality in inputs.items():
            history = self.quality_histories[name]
            history.append(float(quality["quality_score"]))
            trend_result = calculate_trend(history, self.trend_delta)
            state_result = classify_sensor_state(
                quality["quality_score"],
                trend_result["trend"],
                self.high_threshold,
                self.low_threshold,
            )
            sensor_results[name] = {
                **quality,
                "trend": trend_result["trend"],
                "trend_slope": trend_result["slope"],
                **state_result,
            }

        no_active_sensor = calculate_sensor_weights(sensor_results)
        return {
            "csi": sensor_results["csi"],
            "rgb": sensor_results["rgb"],
            "lidar": sensor_results["lidar"],
            "no_active_sensor": no_active_sensor,
        }


# Preserve the established import/API while extending it to all three sensors.
CameraAdaptiveController = SensorQualityController
