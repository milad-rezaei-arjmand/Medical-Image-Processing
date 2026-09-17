import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig0314(a)(100-dollars).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

result = img & 0b11000000

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 3.15(a) Planes 8 and 7")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0315_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0315_result.png")