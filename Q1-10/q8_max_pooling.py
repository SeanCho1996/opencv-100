import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.uint8)

h, w, c = img_rgb.shape
img_pool = np.zeros_like(img_rgb)
for i in range(0, h//8):
    for j in range(0, w//8):
        sub_part = img_rgb[i*8:i*8+8, j*8:j*8+8]
        img_pool[i * 8:i * 8 + 8, j * 8:j * 8 + 8, 0] = np.max(sub_part[:, :, 0])
        img_pool[i * 8:i * 8 + 8, j * 8:j * 8 + 8, 1] = np.max(sub_part[:, :, 1])
        img_pool[i * 8:i * 8 + 8, j * 8:j * 8 + 8, 2] = np.max(sub_part[:, :, 2])
plt.figure()
plt.imshow(img_pool)
plt.show()
