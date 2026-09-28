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


def fourier_spectrum(image):
    spectrum = np.fft.fftshift(
        np.fft.fft2(image)
    )
    magnitude = np.log1p(
        np.abs(spectrum)
    )
    return normalize_to_uint8(magnitude)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the centered Fourier magnitude spectrum."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter4",
                "fourier_spectrum.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = fourier_spectrum(image)

    output = save_comparison(
        image,
        result,
        args.output,
        "Fourier Spectrum",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
