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


def motion_blur_degradation(
    image,
    a=0.1,
    b=0.1,
    exposure=1.0,
):
    a = float(a)
    b = float(b)
    exposure = float(exposure)

    if exposure <= 0:
        raise ValueError(
            "exposure must be greater than zero."
        )

    image_float = image.astype(np.float32)

    rows, cols = image.shape
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    V, U = np.meshgrid(v, u)

    uv = U * a + V * b

    # np.sinc(x) = sin(pi*x)/(pi*x), including the x=0 limit.
    transfer = (
        exposure
        * np.sinc(uv)
        * np.exp(-1j * np.pi * uv)
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
        description="Simulate linear motion-blur degradation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.1)
    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter5",
                "motion_blur.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = motion_blur_degradation(
        image,
        a=args.a,
        b=args.b,
        exposure=args.exposure,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        "Motion Blur Degradation",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
