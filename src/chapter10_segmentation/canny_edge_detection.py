import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig1026(a)(headCT-Vandy).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# طبق خود شکل:
# sigma = 2
# mask size = 13x13
# T_L = 0.05 , T_H = 0.15  --> در بازه 0 تا 255
blurred = cv2.GaussianBlur(img, (13, 13), 2)

low_thresh = int(0.05 * 255)
high_thresh = int(0.15 * 255)

result = cv2.Canny(blurred, low_thresh, high_thresh, L2gradient=True)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 10.26(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 10.26(d) Canny")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig1026_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig1026_result.png")