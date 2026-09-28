"""
Shared I/O and visualization helpers for the image-processing examples.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_grayscale(path):
    """Load an image as 8-bit grayscale."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read the input image: {path}"
        )

    return image


def normalize_to_uint8(image):
    """Normalize a numeric image to the full uint8 range."""

    image = np.asarray(image)

    if image.size == 0:
        raise ValueError("Cannot normalize an empty image.")

    if not np.all(np.isfinite(image)):
        raise ValueError(
            "Processed image contains NaN or infinite values."
        )

    normalized = cv2.normalize(
        image.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return normalized.astype(np.uint8)


def default_output_path(chapter, filename):
    """Return the default generated-output path."""

    return PROJECT_ROOT / "outputs" / chapter / filename


def save_comparison(
    original,
    processed,
    output_path,
    processed_title,
    original_title="Input",
):
    """Save a side-by-side grayscale comparison figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    axis_original = figure.add_subplot(
        1, 2, 1
    )
    axis_original.imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axis_original.set_title(
        original_title
    )
    axis_original.axis("off")

    axis_processed = figure.add_subplot(
        1, 2, 2
    )
    axis_processed.imshow(
        processed,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axis_processed.set_title(
        processed_title
    )
    axis_processed.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path
