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


def median_filter(image, kernel_size=3):
    kernel_size = int(kernel_size)

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd integer greater than 1."
        )

    return cv2.medianBlur(
        image,
        kernel_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply median filtering."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "median_filtering.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = median_filter(
        image,
        args.kernel_size,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Median Filter ({args.kernel_size}x{args.kernel_size})",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
