import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0425(a)(translated_rectangle).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

magnitude = np.abs(fshift)
result = np.log1p(magnitude)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 4.25(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 4.25(b) Fourier Spectrum")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0425_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0425_result.png")