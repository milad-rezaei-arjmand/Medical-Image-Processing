# Medical Image Processing

## Classical Enhancement, Frequency Filtering, Degradation Modeling, Segmentation, and Edge Detection in Python

This repository is an implementation-oriented study of classical digital image-processing methods with applications to biomedical and medical imaging.

It provides reusable command-line scripts for grayscale image enhancement, spatial/frequency processing, degradation modeling, threshold-based segmentation, and edge detection.

The code is intended for education, experimentation, and portfolio demonstration. It is **not** a clinically validated imaging system.

---

## Implemented Methods

### Chapter 3 — Intensity and Spatial Processing

- Image negative
- Logarithmic transformation
- Gamma transformation
- Bit-plane slicing
- High-bit-plane reconstruction
- Histogram equalization
- Median filtering

### Chapter 4 — Frequency-Domain Processing

- Fourier magnitude spectrum
- Ideal low-pass filtering
- Gaussian low-pass filtering
- Ideal high-pass filtering
- Butterworth high-pass filtering
- Homomorphic filtering

### Chapter 5 — Degradation Models

- Atmospheric-turbulence degradation model
- Linear motion-blur degradation model

### Chapter 10 — Segmentation and Edges

- Fixed global thresholding
- Iterative basic global thresholding
- Otsu thresholding
- Canny edge detection

---

## Key Reproducibility Improvement

Older versions of the repository hard-coded textbook-companion filenames such as:

```text
Fig0304(a)(breast_digital_Xray).tif
Fig0462(a)(PET_image).tif
Fig1026(a)(headCT-Vandy).tif
```

Those files were not included in the repository, so a fresh clone could not run the scripts directly.

The cleaned implementation removes hard-coded input filenames. Every script now accepts:

```text
--input
```

and writes generated figures to `outputs/` by default.

This allows the algorithms to run with any compatible grayscale image supplied by the user.

---

## Installation

```bash
git clone https://github.com/milad-rezaei-arjmand/Medical-Image-Processing.git
cd Medical-Image-Processing

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies:

- NumPy
- OpenCV
- Matplotlib

---

## Input Images

The repository does not redistribute the third-party images used by older versions of the scripts.

Place your own input image locally, for example:

```text
data/my_image.png
```

The `data/` directory is ignored by Git except for its README.

See [`data/README.md`](data/README.md).

---

## Usage

### Image negative

```bash
python src/chapter3_intensity_processing/image_negative.py \
  --input data/my_image.png
```

### Gamma transformation

```bash
python src/chapter3_intensity_processing/gamma_transformation.py \
  --input data/my_image.png \
  --gamma 0.6
```

### Gaussian low-pass filter

```bash
python src/chapter4_frequency_processing/gaussian_lowpass_filter.py \
  --input data/my_image.png \
  --cutoff 10
```

### Homomorphic filtering

```bash
python src/chapter4_frequency_processing/homomorphic_filtering.py \
  --input data/my_image.png
```

### Otsu thresholding

```bash
python src/chapter10_segmentation/otsu_thresholding.py \
  --input data/my_image.png
```

### Canny edge detection

```bash
python src/chapter10_segmentation/canny_edge_detection.py \
  --input data/my_image.png \
  --sigma 2 \
  --kernel-size 13 \
  --low-threshold 0.05 \
  --high-threshold 0.15
```

Each script also supports `--help`.

---

## Output Behavior

By default, generated comparison figures are written to:

```text
outputs/chapter3/
outputs/chapter4/
outputs/chapter5/
outputs/chapter10/
```

You can override the destination:

```bash
python src/chapter10_segmentation/otsu_thresholding.py \
  --input data/my_image.png \
  --output outputs/custom_otsu.png
```

Generated outputs are ignored by Git.

See [`results/README.md`](results/README.md) for the repository's result/provenance policy.

---

## Repository Structure

```text
Medical-Image-Processing/
├── data/
│   └── README.md
├── results/
│   └── README.md
├── src/
│   ├── common.py
│   ├── smoke_test.py
│   ├── chapter3_intensity_processing/
│   ├── chapter4_frequency_processing/
│   ├── chapter5_restoration/
│   └── chapter10_segmentation/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Smoke Test

A synthetic-image smoke test validates the core implementations without requiring third-party image files:

```bash
python src/smoke_test.py
```

It checks that all 19 processing methods return finite 2-D outputs with the expected image shape.

---

## Technical Notes

- Input is loaded as 8-bit grayscale.
- Frequency-domain scripts use NumPy FFT operations.
- Frequency-filter outputs are normalized to 8-bit only for visualization.
- The homomorphic filter uses a log-domain illumination/reflectance model.
- The degradation scripts simulate image formation effects; they are not inverse-restoration algorithms.
- Canny thresholds are expressed as fractions of the 8-bit intensity range.
- The methods are classical image-processing demonstrations rather than learned AI models.

---

## Reference

The chapter organization and many of the classical methods are based on concepts commonly presented in:

Rafael C. Gonzalez and Richard E. Woods, *Digital Image Processing*, Pearson.

The book is used as a conceptual reference. Companion/source images from the book are not distributed in this cleaned repository.

---

## License

Repository code and documentation are released under the MIT License.

External input images, datasets, and other third-party materials are not covered by this repository license and remain subject to their own terms.

---

## Author

**Milad Rezaei Arjmand**

M.Sc. Student in Biomedical Engineering (Bioelectric)

Research interests include Medical AI, Biomedical Image Processing, Medical Imaging, and Biomedical Signal Processing.
