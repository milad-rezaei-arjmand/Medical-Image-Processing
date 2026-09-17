import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0526(a)(original_DIP).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

a = 0.1
b = 0.1
T = 1

img_float = img.astype(np.float32)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

u = np.arange(rows) - crow
v = np.arange(cols) - ccol
V, U = np.meshgrid(v, u)

uv = U * a + V * b

# Motion blur degradation function:
# H(u,v) = T / [pi(ua+vb)] * sin[pi(ua+vb)] * exp[-j*pi(ua+vb)]
H = np.zeros((rows, cols), dtype=np.complex64)

pi_uv = np.pi * uv

# جلوگیری از تقسیم بر صفر
mask = pi_uv != 0

H[mask] = (T / pi_uv[mask]) * np.sin(pi_uv[mask]) * np.exp(-1j * pi_uv[mask])
H[~mask] = T

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
plt.title("Fig 5.26(a) Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 5.26(b) Motion Blur")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0526_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0526_result.png")