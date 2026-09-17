import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0441(a)(characters_test_pattern).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

D0 = 30
n = 2

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

u = np.arange(rows)
v = np.arange(cols)
V, U = np.meshgrid(v, u)

D = np.sqrt((U - crow) ** 2 + (V - ccol) ** 2)

D[D == 0] = 1e-6
H = 1 / (1 + (D0 / D) ** (2 * n))

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

gshift = fshift * H

g = np.fft.ifftshift(gshift)
result = np.fft.ifft2(g)
result = np.real(result)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 4.55(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 4.55(b) BHPF D0 = 30")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0455_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0455_result.png")