import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Fig0335(a)(ckt_board_saltpep_prob_pt05).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

result = cv2.medianBlur(img, 3)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 3.35(a) Original Noisy")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 3.35(c) 3x3 Median")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0335_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0335_result.png")