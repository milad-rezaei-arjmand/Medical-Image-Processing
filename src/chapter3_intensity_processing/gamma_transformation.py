import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0308(a)(fractured_spine).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

gamma = 0.6

img_norm = img.astype(np.float32) / 255.0
result = np.power(img_norm, gamma)
result = np.uint8(result * 255)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 3.8(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 3.8(b) Gamma = 0.6")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0308_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0308_result.png")