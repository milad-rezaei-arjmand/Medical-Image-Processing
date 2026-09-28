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


def basic_global_threshold(
    image,
    initial_threshold=None,
    tolerance=0.5,
    max_iterations=100,
):
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be greater than zero."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    threshold = (
        float(np.mean(image))
        if initial_threshold is None
        else float(initial_threshold)
    )

    if not 0 <= threshold <= 255:
        raise ValueError(
            "initial_threshold must be in [0, 255]."
        )

    for _ in range(max_iterations):
        group_high = image[image > threshold]
        group_low = image[image <= threshold]

        if len(group_high) == 0 or len(group_low) == 0:
            break

        new_threshold = 0.5 * (
            float(group_high.mean())
            + float(group_low.mean())
        )

        if abs(threshold - new_threshold) < tolerance:
            threshold = new_threshold
            break

        threshold = new_threshold

    _, result = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return threshold, result


def main():
    parser = argparse.ArgumentParser(
        description="Apply iterative basic global thresholding."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--initial-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter10",
                "basic_global_thresholding.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    threshold, result = basic_global_threshold(
        image,
        initial_threshold=args.initial_threshold,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Basic Global Threshold (T={threshold:.2f})",
    )
    print(f"Threshold: {threshold:.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
