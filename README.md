# Medical Image Processing:

## Enhancement, Filtering, Restoration, and Segmentation

## Overview

This repository presents an implementation-based study of classical
medical image processing techniques using Python.

The project focuses on fundamental digital image processing methods
including image enhancement, spatial and frequency domain processing,
image restoration, and image segmentation.

The implementations are based on concepts from the book:

**Digital Image Processing**\
Rafael C. Gonzalez and Richard E. Woods

------------------------------------------------------------------------

# Implemented Methods

## Chapter 3 --- Image Enhancement and Spatial Processing

Implemented techniques:

-   Image Negative Transformation
-   Logarithmic Transformation
-   Gamma Transformation
-   Bit-Plane Slicing
-   High Bit-Plane Reconstruction
-   Histogram Equalization
-   Median Filtering

## Chapter 4 --- Frequency Domain Processing

Implemented techniques:

-   Fourier Spectrum Analysis
-   Ideal Low-Pass Filtering
-   Gaussian Low-Pass Filtering
-   Ideal High-Pass Filtering
-   Butterworth High-Pass Filtering
-   Homomorphic Filtering

## Chapter 5 --- Image Restoration

Implemented techniques:

-   Atmospheric Turbulence Degradation Model
-   Linear Motion Blur Degradation Model

## Chapter 10 --- Image Segmentation

Implemented techniques:

-   Global Thresholding
-   Basic Global Thresholding
-   Otsu Thresholding
-   Canny Edge Detection

------------------------------------------------------------------------

# Technologies

Programming Language:

-   Python

Libraries:

-   OpenCV
-   NumPy
-   Matplotlib

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/milad-rezaei-arjmand/Medical-Image-Processing.git
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

Each implementation can be executed independently.

Example:

``` bash
python src/chapter10_segmentation/otsu_thresholding.py
```

Generated outputs are saved inside the `results` directory.

------------------------------------------------------------------------

# Results

The repository includes generated visualization results for:

-   Image enhancement
-   Frequency filtering
-   Restoration simulations
-   Segmentation methods

------------------------------------------------------------------------

# Reference

Gonzalez, R. C., & Woods, R. E.

**Digital Image Processing**

Pearson Education.

------------------------------------------------------------------------

# Author

**Milad Rezaei Arjmand**

Research Interests:

-   Medical Artificial Intelligence
-   Biomedical Image Processing
-   Signal and Image Processing
-   Deep Learning
