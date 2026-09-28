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


def butterworth_highpass_filter(
    image,
    cutoff=30.0,
    order=2,
):
    cutoff = float(cutoff)
    order = int(order)

    if cutoff <= 0:
        raise ValueError("cutoff must be greater than zero.")

    if order <= 0:
        raise ValueError("order must be greater than zero.")

    rows, cols = image.shape
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)

    distance = np.sqrt(
        (U - center_row) ** 2
        + (V - center_col) ** 2
    )

    safe_distance = np.maximum(
        distance,
        1e-12,
    )

    transfer = 1.0 / (
        1.0
        + (cutoff / safe_distance) ** (2 * order)
    )

    transfer[distance == 0] = 0.0

    spectrum = np.fft.fftshift(
        np.fft.fft2(image)
    )
    filtered = spectrum * transfer
    result = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(filtered)
        )
    )

    return normalize_to_uint8(result)


def main():
    parser = argparse.ArgumentParser(
        description="Apply a Butterworth high-pass frequency filter."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--cutoff", type=float, default=30.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument(
        "--output",
        default=str(
            default_output_path(
                "chapter4",
                "butterworth_highpass_filter.png",
            )
        ),
    )
    args = parser.parse_args()

    image = load_grayscale(args.input)
    result = butterworth_highpass_filter(
        image,
        cutoff=args.cutoff,
        order=args.order,
    )

    output = save_comparison(
        image,
        result,
        args.output,
        (
            "Butterworth High-Pass "
            f"(D0={args.cutoff:g}, n={args.order})"
        ),
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
