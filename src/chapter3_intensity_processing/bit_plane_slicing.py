import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0314(a)(100-dollars).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

bit_number = 8

result = ((img >> (bit_number - 1)) & 1) * 255
result = result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 3.14(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title(f"Fig 3.14 Bit Plane {bit_number}")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0314_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0314_result.png")