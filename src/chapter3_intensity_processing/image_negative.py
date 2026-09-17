import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig0304(a)(breast_digital_Xray).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

result = 255 - img

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 3.4(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 3.4(b) Negative")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0304_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0304_result.png")