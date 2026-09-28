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


def image_negative(image):
    return (255 - image).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Apply grayscale image-negative transformation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "image_negative.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = image_negative(image)

    output = save_comparison(
        image,
        result,
        args.output,
        "Image Negative",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
