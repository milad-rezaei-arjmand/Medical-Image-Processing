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


def canny_edge_detection(
    image,
    sigma=2.0,
    kernel_size=13,
    low_threshold=0.05,
    high_threshold=0.15,
):
    sigma = float(sigma)
    kernel_size = int(kernel_size)
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd integer greater than 1."
        )

    if not (
        0 <= low_threshold < high_threshold <= 1
    ):
        raise ValueError(
            "Threshold fractions must satisfy "
            "0 <= low < high <= 1."
        )

    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma,
    )

    low_value = int(
        round(low_threshold * 255)
    )
    high_value = int(
        round(high_threshold * 255)
    )

    return cv2.Canny(
        blurred,
        low_value,
        high_value,
        L2gradient=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply Gaussian smoothing followed by Canny edge detection."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.05,
        help="Fraction of the 8-bit range.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.15,
        help="Fraction of the 8-bit range.",
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter10",
                "canny_edge_detection.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = canny_edge_detection(
        image,
        sigma=args.sigma,
        kernel_size=args.kernel_size,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        "Canny Edge Detection",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
