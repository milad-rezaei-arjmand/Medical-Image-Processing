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


def log_transform(image):
    image_float = image.astype(np.float32)
    transformed = np.log1p(image_float)
    return normalize_to_uint8(transformed)


def main():
    parser = argparse.ArgumentParser(
        description="Apply logarithmic intensity transformation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "log_transformation.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = log_transform(image)

    output = save_comparison(
        image,
        result,
        args.output,
        "Log Transformation",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
