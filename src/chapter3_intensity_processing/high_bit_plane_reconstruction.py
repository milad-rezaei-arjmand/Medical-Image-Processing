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


def reconstruct_from_bits(image, bits=(8, 7)):
    bits = tuple(int(bit) for bit in bits)

    if not bits:
        raise ValueError("At least one bit plane must be selected.")

    if any(bit < 1 or bit > 8 for bit in bits):
        raise ValueError("All bit numbers must be between 1 and 8.")

    mask = 0

    for bit in bits:
        mask |= 1 << (bit - 1)

    return np.bitwise_and(
        image,
        np.uint8(mask),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct an image from selected bit planes."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 7],
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "high_bit_plane_reconstruction.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = reconstruct_from_bits(
        image,
        bits=args.bits,
    )

    bit_text = ", ".join(
        str(bit) for bit in args.bits
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Reconstruction from Bits {bit_text}",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
