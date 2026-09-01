# Quality-Adaptive Multi-Sensor Obstacle Fusion for QCar2

This dissertation project investigates obstacle detection using a front CSI
camera, an RGB camera, and LiDAR in Quanser Interactive Labs (QLabs). It
compares a quality-adaptive fusion method with a fixed-weight baseline across
three obstacle layouts and six clear or adverse-weather scenarios.

## System Pipeline

```text
Synchronized sensor acquisition
        -> camera quality assessment
        -> active-sensor selection
        -> active-only weight allocation
        -> selected-sensor detection
        -> bearing-based association
        -> decision-level fusion
        -> offline Ground Truth evaluation
```

In the adaptive method, camera quality is calculated from image sharpness and
contrast. Cameras with a quality score below `0.75` are inactive and do not run
object detection. The remaining sensors receive normalized squared-quality
weights. Associated detections within a `1.0 degree` bearing gate are accepted
when their binary weighted fusion score is at least `0.5`.

The fixed-weight baseline keeps CSI, RGB, and LiDAR active with a weight of
`1/3` each. Ground Truth is used only for offline evaluation and never affects
online sensor selection, detection, association, or fusion.

## Weather Scenarios

| Script | Scenario |
|---|---|
| `S1_Clear.py` | Clear weather |
| `S2_Clear_to_Rain.py` | Clear -> Rain -> Clear |
| `S3_Clear_to_Fog.py` | Clear -> Fog -> Clear |
| `S4_Clear_to_Cloudy.py` | Clear -> Cloudy -> Clear |
| `S5_Clear_to_Blizzard.py` | Clear -> Blizzard -> Clear |
| `S6_Clear_to_Cloudy_to_Thunderstorm.py` | Clear -> Cloudy -> Thunderstorm -> Clear |

Each scenario can be run with obstacle Layout 1, 2, or 3 and with either the
dynamic-weight or fixed-weight fusion method.

## Main Files

| File or directory | Purpose |
|---|---|
| `qcar2.py` | Creates the QLabs scene and coordinates data collection |
| `sensor_data.py` | Runs and records adaptive-weight experiments |
| `sensor_fixed_weight.py` | Runs the fixed-weight baseline |
| `features_extract.py` | Calculates camera image-quality features |
| `camera_adaptive_control.py` | Selects sensors and allocates adaptive weights |
| `detection_results.py` | Runs independent camera and LiDAR detection |
| `lidar_obstacle_detector.py` | Detects obstacles from LiDAR scans |
| `adaptive_object_fusion.py` | Associates detections and performs fusion |
| `obstacle_layout_1.py` to `obstacle_layout_3.py` | Define the three obstacle layouts |
| `Final_Evaluation.py` | Evaluates final obstacles against Ground Truth |
| `YOLO_Dataset/` | Camera obstacle-detection dataset |
| `YOLO_Obstacle_Training/` | Trained model and training results |

## Requirements

- Python 3
- Quanser Interactive Labs and the QVL Python API
- OpenCV, NumPy, PyQtGraph, and a Qt backend
- Ultralytics and PyTorch
- Pandas and Pillow for calibration or model-training utilities

The frozen camera model must exist at:

```text
YOLO_Obstacle_Training/yolo26s_obstacle_v1/weights/best.pt
```

## Running an Experiment

1. Start QLabs and ensure it is available at `localhost`.
2. Open a terminal in the project directory.
3. Run one weather scenario, for example:

```powershell
python S1_Clear.py
```

4. Select the fusion mode when prompted:
   - `1` - Dynamic Weight
   - `2` - Fixed Weight
5. Select obstacle Layout `1`, `2`, or `3`.

Experimental data are saved automatically under:

```text
Dynamic Weight Experimental Data/<scenario>/test<id>_layout<id>/
Fixed Weight Experimental Data/<scenario>/test<id>_layout<id>/
```

Each run contains synchronized CSI images, RGB images, LiDAR scans, camera
quality data, independent detection results, fusion results, and accepted final
obstacles.

## Offline Evaluation

Evaluate all available dynamic- and fixed-weight experiments:

```powershell
python Final_Evaluation.py
```

Evaluation results are written to `Final Evaluation Results/`. The evaluator
uses one-to-one bearing matching to calculate frame-level TP, FP, FN,
precision, recall, and F1-score.

## Tests

Run the detector-free fusion and evaluation tests with:

```powershell
python -m unittest test_adaptive_object_fusion.py test_final_evaluation.py
```

