import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig1038(a)(noisy_fingerprint).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# آستانه‌ی سراسری
T = 127

_, result = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 10.38(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 10.38(c) Global Threshold")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig1038_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig1038_result.png")