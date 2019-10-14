import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('../images/imori_dark.jpg').astype(np.uint8)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# r, g, b = cv.split(img_rgb)
# hist_r = cv.calcHist([r], [0], None, [256], [0, 255])  # 求直方图
# hist_g = cv.calcHist([g], [0], None, [256], [0, 255])
# hist_b = cv.calcHist([b], [0], None, [256], [0, 255])
m0 = np.mean(img_rgb)
sig0 = np.std(img_rgb)
m = 128
sig = 52
img_op = (sig / sig0 * (img_rgb - m0) + m).astype(np.uint8)  # 对直方图操作，实际上就是对原图操作！！

plt.figure()
plt.subplot(1, 2, 1)
plt.hist(img_rgb.ravel(), bins=256, range=[0, 255], rwidth=0.8)
plt.subplot(1, 2, 2)
plt.hist(img_op.ravel(), bins=256, range=[0, 255], rwidth=0.8)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.subplot(1, 2, 2)
plt.imshow(img_op)
plt.show()
