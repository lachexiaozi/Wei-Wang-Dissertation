import re
import warnings
from pathlib import Path

import cv2
import numpy as np


WORKSPACE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CSI_DATA_DIRECTORY = WORKSPACE_DIRECTORY / "Front CSI Camera Data"
DEFAULT_RGB_DATA_DIRECTORY = WORKSPACE_DIRECTORY / "RGB Camera Data"

CSI_IMAGE_PATTERN = re.compile(
    r"Front_CSI_Camera_Image_(\d+)\.jpg$",
    re.IGNORECASE,
)
RGB_IMAGE_PATTERN = re.compile(
    r"RGB_Camera_Image_(\d+)\.jpg$",
    re.IGNORECASE,
)


def _validate_thresholds(bright_threshold, dark_threshold):
    if not isinstance(bright_threshold, int) or not 0 <= bright_threshold <= 255:
        raise ValueError("bright_threshold must be an integer from 0 to 255")
    if not isinstance(dark_threshold, int) or not 0 <= dark_threshold <= 255:
        raise ValueError("dark_threshold must be an integer from 0 to 255")


def calculate_image_features(image, bright_threshold=200, dark_threshold=50):
    """Calculate reproducible image-quality features from an in-memory frame.

    Contrast is the grayscale pixel standard deviation. Sharpness is the
    variance of the grayscale Laplacian (a common focus measure).
    """
    _validate_thresholds(bright_threshold, dark_threshold)
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")

    if image.ndim == 2:
        grayscale_image = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")

    mean_value, standard_deviation = cv2.meanStdDev(grayscale_image)
    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    _, laplacian_standard_deviation = cv2.meanStdDev(laplacian)

    bright_mask = cv2.compare(grayscale_image, bright_threshold, cv2.CMP_GE)
    dark_mask = cv2.compare(grayscale_image, dark_threshold, cv2.CMP_LE)

    return {
        "brightness": float(mean_value[0, 0]),
        "contrast": float(standard_deviation[0, 0]),
        "sharpness": float(laplacian_standard_deviation[0, 0] ** 2),
        "bright_ratio": float(cv2.countNonZero(bright_mask) / grayscale_image.size),
        "dark_ratio": float(cv2.countNonZero(dark_mask) / grayscale_image.size),
    }


def calculate_relative_quality(current_value, baseline):
    """Score how closely a value matches its baseline on either side."""
    if (
        not np.isfinite(current_value)
        or not np.isfinite(baseline)
        or current_value <= 0
        or baseline <= 0
    ):
        return 0.0

    return float(np.clip(min(
        current_value / baseline,
        baseline / current_value,
    ), 0.0, 1.0))


def calculate_camera_quality(
    frame,
    sharpness_baseline,
    contrast_baseline,
):
    """Calculate raw and baseline-normalized quality metrics for one frame."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty OpenCV image")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"unsupported frame shape: {frame.shape}")

    # Keep full precision internally; rounding is performed only for CSV output.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.std(gray))
    sharpness_score = calculate_relative_quality(
        sharpness,
        sharpness_baseline,
    )
    contrast_score = calculate_relative_quality(
        contrast,
        contrast_baseline,
    )
    quality_score = (sharpness_score + contrast_score) / 2.0

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "sharpness_score": sharpness_score,
        "contrast_score": contrast_score,
        "quality_score": quality_score,
    }


def _find_numbered_images(data_source, filename_pattern, camera_name):
    data_source = Path(data_source)
    if data_source.is_file():
        match = filename_pattern.fullmatch(data_source.name)
        if match is None:
            raise ValueError(f"Unexpected {camera_name} filename: {data_source.name}")
        numbered_images = [(int(match.group(1)), data_source)]
    elif data_source.is_dir():
        numbered_images = []
        for image_path in data_source.iterdir():
            if not image_path.is_file():
                continue
            match = filename_pattern.fullmatch(image_path.name)
            if match is not None:
                numbered_images.append((int(match.group(1)), image_path))
    else:
        raise FileNotFoundError(
            f"{camera_name} data source does not exist: {data_source}"
        )

    numbered_images.sort(key=lambda item: item[0])
    return numbered_images


def _extract_camera_features(
    data_source,
    filename_pattern,
    camera_name,
    bright_threshold,
    dark_threshold,
):
    """Extract the shared image features for one camera data source."""
    _validate_thresholds(bright_threshold, dark_threshold)
    numbered_images = _find_numbered_images(
        data_source,
        filename_pattern,
        camera_name,
    )

    extracted_features = []
    for sample_number, image_path in numbered_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            warnings.warn(
                f"Unable to read {camera_name} image; skipped: {image_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        image_features = calculate_image_features(
            image, bright_threshold, dark_threshold
        )

        feature = {
            "sample_number": sample_number,
            "filename": image_path.name,
            **image_features,
        }
        extracted_features.append(feature)
        print(
            f"{feature['filename']} | "
            f"brightness: {feature['brightness']:.2f} | "
            f"contrast: {feature['contrast']:.2f} | "
            f"sharpness: {feature['sharpness']:.2f} | "
            f"bright_ratio: {feature['bright_ratio']:.4f} | "
            f"dark_ratio: {feature['dark_ratio']:.4f}"
        )

    return extracted_features


def extract_CSI_Camera(
    data_source=DEFAULT_CSI_DATA_DIRECTORY,
    bright_threshold=200,
    dark_threshold=50,
):
    """Extract and print features from Front CSI camera images."""
    return _extract_camera_features(
        data_source,
        CSI_IMAGE_PATTERN,
        "Front CSI Camera",
        bright_threshold,
        dark_threshold,
    )


def extract_RGB_Camera(
    data_source=DEFAULT_RGB_DATA_DIRECTORY,
    bright_threshold=200,
    dark_threshold=50,
):
    """Extract and print features from RGB camera images."""
    return _extract_camera_features(
        data_source,
        RGB_IMAGE_PATTERN,
        "RGB Camera",
        bright_threshold,
        dark_threshold,
    )
