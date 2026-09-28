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


def global_threshold(image, threshold=127):
    threshold = float(threshold)

    if not 0 <= threshold <= 255:
        raise ValueError(
            "threshold must be in the range [0, 255]."
        )

    _, result = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Apply fixed global thresholding."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=127.0,
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter10",
                "global_thresholding.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = global_threshold(
        image,
        threshold=args.threshold,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Global Threshold (T={args.threshold:g})",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
