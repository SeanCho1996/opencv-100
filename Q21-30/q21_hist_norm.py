import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_dark.jpg').astype(np.uint8)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_r = cv.equalizeHist(img_rgb[:, : ,0])  # 这里注意equalizeHist函数只能用于单层图像（灰度图像）
img_g = cv.equalizeHist(img_rgb[:, :, 1])
img_b = cv.equalizeHist(img_rgb[:, :, 2])

img_norm = cv.merge([img_r, img_g, img_b])  # 将三个子图像融合成最后的图像
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.subplot(1, 2, 2)
plt.imshow(img_norm)
plt.show()

