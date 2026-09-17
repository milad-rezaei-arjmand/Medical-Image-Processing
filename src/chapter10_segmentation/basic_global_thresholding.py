import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig1039(a)(polymersomes).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Basic Global Thresholding Algorithm
T = img.mean()

while True:
    G1 = img[img > T]
    G2 = img[img <= T]

    if len(G1) == 0 or len(G2) == 0:
        break

    m1 = G1.mean()
    m2 = G2.mean()
    T_new = 0.5 * (m1 + m2)

    if abs(T - T_new) < 0.5:
        break

    T = T_new

_, result = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 10.39(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 10.39(c) Basic Global")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig1039_c_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig1039_c_result.png")