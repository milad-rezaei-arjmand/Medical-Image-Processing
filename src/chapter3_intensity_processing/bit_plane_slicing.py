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


def extract_bit_plane(image, bit_number=8):
    bit_number = int(bit_number)

    if not 1 <= bit_number <= 8:
        raise ValueError("bit_number must be between 1 and 8.")

    plane = (
        (image >> (bit_number - 1)) & 1
    ) * 255

    return plane.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Extract one bit plane from an 8-bit grayscale image."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--bit",
        type=int,
        default=8,
        choices=range(1, 9),
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "bit_plane_slicing.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = extract_bit_plane(
        image,
        args.bit,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Bit Plane {args.bit}",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
