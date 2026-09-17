import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig1039(a)(polymersomes).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

_, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 10.39(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 10.39(d) Otsu")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig1039_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig1039_result.png")