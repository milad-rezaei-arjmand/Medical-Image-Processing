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


def atmospheric_turbulence_degradation(
    image,
    k=0.0025,
):
    k = float(k)

    if k < 0:
        raise ValueError("k must be non-negative.")

    image_float = image.astype(np.float32)

    rows, cols = image.shape
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    V, U = np.meshgrid(v, u)

    distance_squared = U**2 + V**2

    transfer = np.exp(
        -k * (distance_squared ** (5.0 / 6.0))
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(image_float)
    )
    degraded = spectrum * transfer

    result = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(degraded)
        )
    )

    return normalize_to_uint8(result)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate atmospheric-turbulence degradation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=float, default=0.0025)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter5",
                "atmospheric_turbulence.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = atmospheric_turbulence_degradation(
        image,
        k=args.k,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        f"Atmospheric Turbulence (k={args.k:g})",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
