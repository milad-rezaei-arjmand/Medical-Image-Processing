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


def histogram_equalization(image):
    return cv2.equalizeHist(image)


def main():
    parser = argparse.ArgumentParser(
        description="Apply global histogram equalization."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "histogram_equalization.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = histogram_equalization(image)

    output = save_comparison(
        image,
        result,
        args.output,
        "Histogram Equalization",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
