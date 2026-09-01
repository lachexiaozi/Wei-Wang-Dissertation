"""Fine-tune pretrained YOLO26s as a single-class QLabs obstacle detector."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# Keep Ultralytics runtime settings with this experiment rather than relying on
# a machine-specific user profile directory.
_ULTRALYTICS_CONFIG_DIRECTORY = (
    Path("YOLO_Obstacle_Training") / ".ultralytics_config"
).resolve()
_ULTRALYTICS_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(_ULTRALYTICS_CONFIG_DIRECTORY),
)

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
import ultralytics


DATASET_ROOT = Path("YOLO_Dataset")
DATA_YAML = DATASET_ROOT / "data.yaml"
PROJECT_DIRECTORY = Path("YOLO_Obstacle_Training")
RUN_NAME = "yolo26s_obstacle_v1"
RUN_DIRECTORY = PROJECT_DIRECTORY / RUN_NAME
BASE_MODEL = "yolo26s.pt"

EXPECTED_TRAIN_IMAGES = 34
EXPECTED_VAL_IMAGES = 10
EXPECTED_TRAIN_LABELS = 34
EXPECTED_VAL_LABELS = 10

EPOCHS = 50
PATIENCE = 20
IMGSZ = 640
BATCH = 8
LR0 = 0.001
MOSAIC = 0.5
MIXUP = 0.0
WORKERS = 0
SEED = 42
DETERMINISTIC = True

VALIDATION_CONF_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILENAME_PATTERN = re.compile(r"^QCar1_(Front_CSI|RGB)_(\d+)$")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_labels(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )


@contextmanager
def working_directory(directory: Path):
    """Temporarily use a directory without changing any dataset files."""
    previous_directory = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)


def check_dataset() -> tuple[list[Path], list[Path]]:
    required_directories = {
        "Train images": DATASET_ROOT / "images" / "train",
        "Validation images": DATASET_ROOT / "images" / "val",
        "Train labels": DATASET_ROOT / "labels" / "train",
        "Validation labels": DATASET_ROOT / "labels" / "val",
    }

    errors: list[str] = []
    if not DATASET_ROOT.is_dir():
        errors.append(f"Dataset root does not exist: {DATASET_ROOT}")
    if not DATA_YAML.is_file():
        errors.append(f"Data YAML does not exist: {DATA_YAML}")
    for description, directory in required_directories.items():
        if not directory.is_dir():
            errors.append(f"{description} directory does not exist: {directory}")

    if errors:
        raise RuntimeError("Dataset check failed:\n- " + "\n- ".join(errors))

    train_images = list_images(required_directories["Train images"])
    val_images = list_images(required_directories["Validation images"])
    train_labels = list_labels(required_directories["Train labels"])
    val_labels = list_labels(required_directories["Validation labels"])

    actual_counts = {
        "Train images": len(train_images),
        "Validation images": len(val_images),
        "Train labels": len(train_labels),
        "Validation labels": len(val_labels),
    }
    expected_counts = {
        "Train images": EXPECTED_TRAIN_IMAGES,
        "Validation images": EXPECTED_VAL_IMAGES,
        "Train labels": EXPECTED_TRAIN_LABELS,
        "Validation labels": EXPECTED_VAL_LABELS,
    }
    for description, expected in expected_counts.items():
        actual = actual_counts[description]
        if actual != expected:
            errors.append(f"{description}: expected {expected}, found {actual}")

    for subset, images, labels in (
        ("train", train_images, train_labels),
        ("val", val_images, val_labels),
    ):
        image_basenames = {path.stem for path in images}
        label_basenames = {path.stem for path in labels}
        missing_labels = sorted(image_basenames - label_basenames)
        orphan_labels = sorted(label_basenames - image_basenames)
        if missing_labels:
            errors.append(f"{subset} missing labels: {', '.join(missing_labels)}")
        if orphan_labels:
            errors.append(f"{subset} orphan labels: {', '.join(orphan_labels)}")

    if errors:
        raise RuntimeError("Dataset check failed:\n- " + "\n- ".join(errors))

    print("========================================")
    print("Dataset Check")
    print("========================================")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Data YAML: {DATA_YAML}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Validation labels: {len(val_labels)}")
    print("Status: PASS")
    print("========================================")
    return train_images, val_images


def camera_and_config(image_path: Path) -> tuple[str, int]:
    match = FILENAME_PATTERN.fullmatch(image_path.stem)
    if not match:
        raise ValueError(f"Unexpected validation filename: {image_path.name}")
    camera = "CSI" if match.group(1) == "Front_CSI" else "RGB"
    return camera, int(match.group(2))


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def make_training_config(device: int | str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device": device,
        "base_model": BASE_MODEL,
        "dataset": str(DATASET_ROOT),
        "data_yaml": str(DATA_YAML),
        "train_image_count": EXPECTED_TRAIN_IMAGES,
        "validation_image_count": EXPECTED_VAL_IMAGES,
        "class_names": {"0": "Obstacle"},
        "training_parameters": {
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "lr0": LR0,
            "mosaic": MOSAIC,
            "mixup": MIXUP,
            "workers": WORKERS,
            "seed": SEED,
            "deterministic": DETERMINISTIC,
            "device": device,
        },
    }


def completed_epochs(run_directory: Path, training_result: Any) -> int:
    results_csv = run_directory / "results.csv"
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
            return sum(1 for _ in csv.DictReader(file))

    trainer = getattr(training_result, "trainer", None)
    epoch = getattr(trainer, "epoch", None)
    if isinstance(epoch, int):
        return epoch + 1
    return 0


def first_numeric_attribute(objects: list[Any], names: list[str]) -> float | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    return None


def extract_ultralytics_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    results_dict = getattr(metrics, "results_dict", {}) or {}

    def dictionary_value(keys: list[str]) -> float | None:
        for key in keys:
            if key in results_dict:
                try:
                    return float(results_dict[key])
                except (TypeError, ValueError):
                    pass
        return None

    return {
        "precision": first_numeric_attribute([box, metrics], ["mp", "precision"])
        or dictionary_value(["metrics/precision(B)", "metrics/precision"]),
        "recall": first_numeric_attribute([box, metrics], ["mr", "recall"])
        or dictionary_value(["metrics/recall(B)", "metrics/recall"]),
        "map50": first_numeric_attribute([box, metrics], ["map50"])
        or dictionary_value(["metrics/mAP50(B)", "metrics/mAP50"]),
        "map50_95": first_numeric_attribute([box, metrics], ["map", "map50_95"])
        or dictionary_value(["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
    }


def load_ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label at {label_path}:{line_number}")
        class_id, x_center, y_center, box_width, box_height = map(float, parts)
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class at {label_path}:{line_number}")
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[list[float]], predictions: list[list[float]]
) -> tuple[int, int, int]:
    candidates = []
    for ground_truth_id, gt_box in enumerate(ground_truth):
        for prediction_id, prediction_box in enumerate(predictions):
            iou = box_iou(gt_box, prediction_box)
            if iou >= MATCH_IOU_THRESHOLD:
                candidates.append((iou, ground_truth_id, prediction_id))

    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for _, ground_truth_id, prediction_id in sorted(candidates, reverse=True):
        if (
            ground_truth_id not in matched_ground_truth
            and prediction_id not in matched_predictions
        ):
            matched_ground_truth.add(ground_truth_id)
            matched_predictions.add(prediction_id)

    true_positives = len(matched_ground_truth)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def safe_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: tuple[int, int, int],
    text: str,
    line_width: int = 3,
) -> None:
    draw.rectangle(tuple(box), outline=color, width=line_width)
    text_box = draw.textbbox((box[0], box[1]), text)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_y = max(0.0, box[1] - text_height - 4)
    draw.rectangle(
        (box[0], text_y, box[0] + text_width + 4, text_y + text_height + 4),
        fill=color,
    )
    draw.text((box[0] + 2, text_y + 2), text, fill="white")


def save_prediction_image(
    image_path: Path,
    output_path: Path,
    predictions: list[list[float]],
    confidences: list[float],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, confidence in zip(predictions, confidences):
        draw_box(draw, box, (220, 20, 60), f"Obstacle {confidence:.2f}")
    image.save(output_path)


def save_comparison_image(
    image_path: Path,
    output_path: Path,
    ground_truth: list[list[float]],
    predictions: list[list[float]],
    confidences: list[float],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 150, 55), fill=(0, 0, 0))
    draw.line((16, 22, 38, 22), fill=(0, 200, 0), width=4)
    draw.text((44, 15), "Ground Truth", fill="white")
    draw.line((16, 43, 38, 43), fill=(220, 20, 60), width=4)
    draw.text((44, 36), "Prediction", fill="white")
    for box in ground_truth:
        draw_box(draw, box, (0, 200, 0), "GT")
    for box, confidence in zip(predictions, confidences):
        draw_box(draw, box, (220, 20, 60), f"Obstacle {confidence:.2f}")
    image.save(output_path)


def aggregate_metrics(rows: list[dict[str, Any]], camera: str | None) -> dict[str, Any]:
    selected = rows if camera is None else [row for row in rows if row["Camera"] == camera]
    tp = sum(row["TP"] for row in selected)
    fp = sum(row["FP"] for row in selected)
    fn = sum(row["FN"] for row in selected)
    precision, recall, f1 = safe_metrics(tp, fp, fn)
    return {
        "images": len(selected),
        "GT": sum(row["GT_Count"] for row in selected),
        "predictions": sum(row["Prediction_Count"] for row in selected),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_custom_validation(
    model: YOLO, val_images: list[Path], device: int | str
) -> dict[str, Any]:
    predictions_directory = RUN_DIRECTORY / "validation_predictions"
    comparison_directory = RUN_DIRECTORY / "validation_comparison"
    predictions_directory.mkdir(parents=True, exist_ok=True)
    comparison_directory.mkdir(parents=True, exist_ok=True)

    detection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    for image_path in val_images:
        camera, config_id = camera_and_config(image_path)
        with Image.open(image_path) as source_image:
            image_width, image_height = source_image.size

        prediction_result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=VALIDATION_CONF_THRESHOLD,
            device=device,
            verbose=False,
        )[0]

        if prediction_result.boxes is None:
            prediction_boxes: list[list[float]] = []
            confidences: list[float] = []
            class_ids: list[int] = []
        else:
            prediction_boxes = prediction_result.boxes.xyxy.cpu().tolist()
            confidences = prediction_result.boxes.conf.cpu().tolist()
            class_ids = [int(value) for value in prediction_result.boxes.cls.cpu().tolist()]

        for detection_id, (box, confidence, class_id) in enumerate(
            zip(prediction_boxes, confidences, class_ids), start=1
        ):
            x1, y1, x2, y2 = box
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            box_area = box_width * box_height
            detection_rows.append(
                {
                    "Filename": image_path.name,
                    "Camera": camera,
                    "Config_ID": config_id,
                    "Detection_ID": detection_id,
                    "Class_ID": class_id,
                    "Class_Name": "Obstacle",
                    "Confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "Box_Width": box_width,
                    "Box_Height": box_height,
                    "Box_Area": box_area,
                    "Box_Area_Ratio": box_area / (image_width * image_height),
                }
            )

        summary_rows.append(
            {
                "Filename": image_path.name,
                "Camera": camera,
                "Config_ID": config_id,
                "Detection_Count": len(confidences),
                "Max_Confidence": max(confidences, default=0.0),
                "Mean_Confidence": mean(confidences) if confidences else 0.0,
                "Min_Confidence": min(confidences, default=0.0),
            }
        )

        label_path = DATASET_ROOT / "labels" / "val" / f"{image_path.stem}.txt"
        ground_truth = load_ground_truth(label_path, image_width, image_height)
        tp, fp, fn = greedy_match(ground_truth, prediction_boxes)
        precision, recall, f1 = safe_metrics(tp, fp, fn)
        per_image_rows.append(
            {
                "Filename": image_path.name,
                "Camera": camera,
                "Config_ID": config_id,
                "GT_Count": len(ground_truth),
                "Prediction_Count": len(prediction_boxes),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        save_prediction_image(
            image_path,
            predictions_directory / image_path.name,
            prediction_boxes,
            confidences,
        )
        save_comparison_image(
            image_path,
            comparison_directory / image_path.name,
            ground_truth,
            prediction_boxes,
            confidences,
        )

    csv_outputs = (
        (RUN_DIRECTORY / "validation_detections.csv", detection_rows),
        (RUN_DIRECTORY / "validation_detection_summary.csv", summary_rows),
        (RUN_DIRECTORY / "validation_per_image_metrics.csv", per_image_rows),
    )
    fieldnames = {
        "validation_detections.csv": [
            "Filename", "Camera", "Config_ID", "Detection_ID", "Class_ID",
            "Class_Name", "Confidence", "x1", "y1", "x2", "y2",
            "Box_Width", "Box_Height", "Box_Area", "Box_Area_Ratio",
        ],
        "validation_detection_summary.csv": [
            "Filename", "Camera", "Config_ID", "Detection_Count",
            "Max_Confidence", "Mean_Confidence", "Min_Confidence",
        ],
        "validation_per_image_metrics.csv": [
            "Filename", "Camera", "Config_ID", "GT_Count", "Prediction_Count",
            "TP", "FP", "FN", "Precision", "Recall", "F1",
        ],
    }
    for path, rows in csv_outputs:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames[path.name])
            writer.writeheader()
            writer.writerows(rows)

    overall = {
        "matching_iou_threshold": MATCH_IOU_THRESHOLD,
        "inference_conf_threshold": VALIDATION_CONF_THRESHOLD,
        "CSI": aggregate_metrics(per_image_rows, "CSI"),
        "RGB": aggregate_metrics(per_image_rows, "RGB"),
        "Combined": aggregate_metrics(per_image_rows, None),
    }
    write_json(RUN_DIRECTORY / "validation_overall_metrics.json", overall)
    return overall


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def print_group_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(f"{name}:")
    print(f"GT = {metrics['GT']}")
    print(f"Predictions = {metrics['predictions']}")
    print(f"TP = {metrics['TP']}")
    print(f"FP = {metrics['FP']}")
    print(f"FN = {metrics['FN']}")
    print(f"Precision = {metrics['precision']:.6f}")
    print(f"Recall = {metrics['recall']:.6f}")
    print(f"F1 = {metrics['f1']:.6f}")
    print()


def main() -> None:
    _, val_images = check_dataset()

    dataset_root_absolute = DATASET_ROOT.resolve()
    data_yaml_absolute = DATA_YAML.resolve()
    project_directory_absolute = PROJECT_DIRECTORY.resolve()
    run_directory_absolute = RUN_DIRECTORY.resolve()

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU may be slow.")

    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIRECTORY / "training_config.json", make_training_config(device))

    model = YOLO(BASE_MODEL)
    # Ultralytics 8.4 resolves YAML `path: .` from the process working
    # directory. Use the dataset root only for the library call; no dataset
    # files are changed.
    with working_directory(dataset_root_absolute):
        training_result = model.train(
            data=str(data_yaml_absolute),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMGSZ,
            batch=BATCH,
            lr0=LR0,
            mosaic=MOSAIC,
            mixup=MIXUP,
            workers=WORKERS,
            seed=SEED,
            deterministic=DETERMINISTIC,
            plots=True,
            save=True,
            verbose=True,
            project=str(project_directory_absolute),
            name=RUN_NAME,
            exist_ok=True,
            device=device,
        )

    best_weights = RUN_DIRECTORY / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best weights: {best_weights}")

    actual_epochs = completed_epochs(RUN_DIRECTORY, training_result)
    early_stopped = actual_epochs < EPOCHS
    best_model = YOLO(str(best_weights))
    with working_directory(dataset_root_absolute):
        validation_result = best_model.val(
            data=str(data_yaml_absolute),
            imgsz=IMGSZ,
            device=device,
            plots=True,
            project=str(run_directory_absolute),
            name="validation",
            exist_ok=True,
        )
    ultralytics_metrics = extract_ultralytics_metrics(validation_result)
    custom_metrics = run_custom_validation(best_model, val_images, device)

    validation_summary = {
        "base_model": BASE_MODEL,
        "best_weights": str(best_weights),
        "dataset": str(DATASET_ROOT),
        "data_yaml": str(DATA_YAML),
        "train_images": EXPECTED_TRAIN_IMAGES,
        "val_images": EXPECTED_VAL_IMAGES,
        "classes": {"0": "Obstacle"},
        "epochs_requested": EPOCHS,
        "actual_epochs_completed": actual_epochs,
        "early_stopped": early_stopped,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "lr0": LR0,
        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "seed": SEED,
        "device": device,
        "validation_conf_threshold": VALIDATION_CONF_THRESHOLD,
        "matching_iou_threshold": MATCH_IOU_THRESHOLD,
        "ultralytics_precision": ultralytics_metrics["precision"],
        "ultralytics_recall": ultralytics_metrics["recall"],
        "mAP50": ultralytics_metrics["map50"],
        "mAP50_95": ultralytics_metrics["map50_95"],
    }
    write_json(RUN_DIRECTORY / "validation_summary.json", validation_summary)

    print("========================================")
    print("YOLO26s Obstacle Fine-Tuning Complete")
    print("========================================")
    print(f"Dataset:\n{DATASET_ROOT}\n")
    print(f"Data YAML:\n{DATA_YAML}\n")
    print(f"Model:\n{BASE_MODEL}\n")
    print(f"Fine-tuned weights:\n{best_weights}\n")
    print("Dataset split:")
    print(f"Train images = {EXPECTED_TRAIN_IMAGES}")
    print(f"Validation images = {EXPECTED_VAL_IMAGES}\n")
    print("Training:")
    print(f"Requested epochs = {EPOCHS}")
    print(f"Actual epochs = {actual_epochs}")
    print(f"Early stopped = {'Yes' if early_stopped else 'No'}\n")
    print("Ultralytics Validation:")
    print(f"Precision = {format_metric(ultralytics_metrics['precision'])}")
    print(f"Recall = {format_metric(ultralytics_metrics['recall'])}")
    print(f"mAP50 = {format_metric(ultralytics_metrics['map50'])}")
    print(f"mAP50-95 = {format_metric(ultralytics_metrics['map50_95'])}\n")
    print(f"Custom Validation\n(conf = {VALIDATION_CONF_THRESHOLD}, IoU = {MATCH_IOU_THRESHOLD}):\n")
    print_group_metrics("Combined", custom_metrics["Combined"])
    print_group_metrics("CSI", custom_metrics["CSI"])
    print_group_metrics("RGB", custom_metrics["RGB"])
    print("Outputs:")
    for output in (
        "best.pt", "last.pt", "results.csv", "results.png",
        "validation_summary.json", "training_config.json",
        "validation_detections.csv", "validation_detection_summary.csv",
        "validation_per_image_metrics.csv", "validation_overall_metrics.json",
        "validation_predictions/", "validation_comparison/",
    ):
        print(output)
    print("========================================")


if __name__ == "__main__":
    main()
