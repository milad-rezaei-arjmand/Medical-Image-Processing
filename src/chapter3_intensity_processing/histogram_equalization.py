import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig0320(1)(top_left).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

result = cv2.equalizeHist(img)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 3.20 Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 3.20 Equalized")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0320_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0320_result.png")