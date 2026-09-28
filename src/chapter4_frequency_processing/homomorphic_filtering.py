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


def homomorphic_filter(
    image,
    gamma_low=0.25,
    gamma_high=2.0,
    c=1.0,
    cutoff=80.0,
):
    gamma_low = float(gamma_low)
    gamma_high = float(gamma_high)
    c = float(c)
    cutoff = float(cutoff)

    if gamma_low <= 0 or gamma_high <= 0:
        raise ValueError("Gamma values must be positive.")

    if gamma_high <= gamma_low:
        raise ValueError(
            "gamma_high must be greater than gamma_low."
        )

    if c <= 0 or cutoff <= 0:
        raise ValueError(
            "c and cutoff must be greater than zero."
        )

    image_float = image.astype(np.float32)
    log_image = np.log1p(image_float)

    rows, cols = image.shape
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)

    distance_squared = (
        (U - center_row) ** 2
        + (V - center_col) ** 2
    )

    transfer = (
        (gamma_high - gamma_low)
        * (
            1.0
            - np.exp(
                -c
                * distance_squared
                / (cutoff**2)
            )
        )
        + gamma_low
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(log_image)
    )
    filtered = spectrum * transfer

    result = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(filtered)
        )
    )
    result = np.expm1(result)

    return normalize_to_uint8(result)


def main():
    parser = argparse.ArgumentParser(
        description="Apply homomorphic filtering."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--gamma-low", type=float, default=0.25)
    parser.add_argument("--gamma-high", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=80.0)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter4",
                "homomorphic_filtering.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = homomorphic_filter(
        image,
        gamma_low=args.gamma_low,
        gamma_high=args.gamma_high,
        c=args.c,
        cutoff=args.cutoff,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        "Homomorphic Filtering",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
