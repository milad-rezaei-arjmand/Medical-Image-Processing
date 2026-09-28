"""
Synthetic smoke test for all tracked image-processing methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.chapter3_intensity_processing.bit_plane_slicing import extract_bit_plane
from src.chapter3_intensity_processing.gamma_transformation import gamma_transform
from src.chapter3_intensity_processing.high_bit_plane_reconstruction import reconstruct_from_bits
from src.chapter3_intensity_processing.histogram_equalization import histogram_equalization
from src.chapter3_intensity_processing.image_negative import image_negative
from src.chapter3_intensity_processing.log_transformation import log_transform
from src.chapter3_intensity_processing.median_filtering import median_filter
from src.chapter4_frequency_processing.butterworth_highpass_filter import butterworth_highpass_filter
from src.chapter4_frequency_processing.fourier_spectrum import fourier_spectrum
from src.chapter4_frequency_processing.gaussian_lowpass_filter import gaussian_lowpass_filter
from src.chapter4_frequency_processing.homomorphic_filtering import homomorphic_filter
from src.chapter4_frequency_processing.ideal_highpass_filter import ideal_highpass_filter
from src.chapter4_frequency_processing.ideal_lowpass_filter import ideal_lowpass_filter
from src.chapter5_restoration.atmospheric_turbulence_model import atmospheric_turbulence_degradation
from src.chapter5_restoration.motion_blur_degradation import motion_blur_degradation
from src.chapter10_segmentation.basic_global_thresholding import basic_global_threshold
from src.chapter10_segmentation.canny_edge_detection import canny_edge_detection
from src.chapter10_segmentation.global_thresholding import global_threshold
from src.chapter10_segmentation.otsu_thresholding import otsu_threshold


def make_synthetic_image(size=256):
    y, x = np.mgrid[0:size, 0:size]

    gradient = (x / max(size - 1, 1)) * 120.0

    circle = (
        (x - size * 0.5) ** 2
        + (y - size * 0.5) ** 2
        < (size * 0.23) ** 2
    )

    rectangle = (
        (x > size * 0.12)
        & (x < size * 0.36)
        & (y > size * 0.15)
        & (y < size * 0.38)
    )

    image = gradient
    image = image + circle * 90.0
    image = image + rectangle * 60.0

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def assert_image(name, output, expected_shape):
    output = np.asarray(output)

    if output.shape != expected_shape:
        raise AssertionError(
            f"{name}: expected shape {expected_shape}, "
            f"got {output.shape}"
        )

    if output.ndim != 2:
        raise AssertionError(
            f"{name}: expected 2-D grayscale output."
        )

    if not np.all(np.isfinite(output)):
        raise AssertionError(
            f"{name}: output contains NaN/Inf."
        )

    print(f"[PASS] {name}")


def main():
    image = make_synthetic_image()

    tests = [
        ("image_negative", image_negative(image)),
        ("log_transform", log_transform(image)),
        ("gamma_transform", gamma_transform(image)),
        ("bit_plane_slicing", extract_bit_plane(image)),
        ("high_bit_plane_reconstruction", reconstruct_from_bits(image)),
        ("histogram_equalization", histogram_equalization(image)),
        ("median_filtering", median_filter(image)),
        ("fourier_spectrum", fourier_spectrum(image)),
        ("ideal_lowpass_filter", ideal_lowpass_filter(image)),
        ("gaussian_lowpass_filter", gaussian_lowpass_filter(image)),
        ("ideal_highpass_filter", ideal_highpass_filter(image)),
        ("butterworth_highpass_filter", butterworth_highpass_filter(image)),
        ("homomorphic_filtering", homomorphic_filter(image)),
        (
            "atmospheric_turbulence",
            atmospheric_turbulence_degradation(image),
        ),
        (
            "motion_blur",
            motion_blur_degradation(image),
        ),
        ("global_thresholding", global_threshold(image)),
        (
            "basic_global_thresholding",
            basic_global_threshold(image)[1],
        ),
        ("otsu_thresholding", otsu_threshold(image)[1]),
        ("canny_edge_detection", canny_edge_detection(image)),
    ]

    for name, output in tests:
        assert_image(
            name,
            output,
            image.shape,
        )

    print("\nALL MEDICAL IMAGE PROCESSING SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
