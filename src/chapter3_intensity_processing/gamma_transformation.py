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


def gamma_transform(image, gamma=0.6):
    gamma = float(gamma)

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    normalized = image.astype(np.float32) / 255.0
    transformed = np.power(normalized, gamma)

    return np.clip(
        transformed * 255.0,
        0,
        255,
    ).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Apply power-law (gamma) transformation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter3",
                "gamma_transformation.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = gamma_transform(
        image,
        gamma=args.gamma,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Gamma Transformation (gamma={args.gamma:g})",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
