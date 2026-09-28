# Input Images

The repository does **not** redistribute the third-party example images referenced by older versions of the scripts.

Use images that you are legally permitted to process and redistribute.

## Recommended Local Layout

```text
data/
├── README.md
└── my_image.png
```

Files placed under `data/` are ignored by Git, except for this README.

The scripts load input images as 8-bit grayscale using OpenCV.

Example:

```bash
python src/chapter3_intensity_processing/image_negative.py \
  --input data/my_image.png
```

Generated comparison figures are written under `outputs/` by default.

## Medical Imaging Note

The examples operate on conventional grayscale raster images.

They do not currently provide dedicated DICOM/NIfTI readers, patient-metadata handling, or clinical validation. For medical research data, de-identification, dataset licensing, and institutional/ethical requirements remain the responsibility of the dataset user.
