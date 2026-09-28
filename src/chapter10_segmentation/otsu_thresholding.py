from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    default_output_path,
    load_grayscale,
    normalize_to_uint8,
    save_comparison,
)


def otsu_threshold(image):
    threshold, result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return float(threshold), result


def main():
    parser = argparse.ArgumentParser(
        description="Apply Otsu automatic thresholding."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter10",
                "otsu_thresholding.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    threshold, result = otsu_threshold(image)

    output = save_comparison(
        image,
        result,
        args.output,
        f"Otsu Threshold (T={threshold:.2f})",
    )
    print(f"Threshold: {threshold:.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
