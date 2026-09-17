import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fig0462(a)(PET_image).tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

gamma_l = 0.25
gamma_h = 2.0
c = 1.0
D0 = 80

img_float = img.astype(np.float32)

img_log = np.log1p(img_float)

f = np.fft.fft2(img_log)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

u = np.arange(rows)
v = np.arange(cols)
V, U = np.meshgrid(v, u)

D2 = (U - crow) ** 2 + (V - ccol) ** 2

H = (gamma_h - gamma_l) * (1 - np.exp(-c * D2 / (D0 ** 2))) + gamma_l

gshift = fshift * H

g = np.fft.ifftshift(gshift)
result = np.fft.ifft2(g)
result = np.real(result)

result = np.expm1(result)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Fig 4.62(a) Original PET")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Fig 4.62(b) Homomorphic")
plt.axis("off")

plt.tight_layout()
plt.savefig("Fig0462_result.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Output saved as Fig0462_result.png")