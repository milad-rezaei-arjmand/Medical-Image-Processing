import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0525(a)(aerial_view_no_turb).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# Fig 5.25(b): severe turbulence
k = 0.0025

img_float = img.astype(np.float32)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

u = np.arange(rows) - crow
v = np.arange(cols) - ccol
V, U = np.meshgrid(v, u)

D2 = U**2 + V**2

# Atmospheric turbulence degradation function:
# H(u,v) = exp[-k * (u^2 + v^2)^(5/6)]
H = np.exp(-k * (D2 ** (5 / 6)))

F = np.fft.fft2(img_float)
F_shift = np.fft.fftshift(F)

G_shift = F_shift * H

G = np.fft.ifftshift(G_shift)
result = np.fft.ifft2(G)
result = np.real(result)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 5.25(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 5.25(b) k = 0.0025")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0525_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0525_result.png")