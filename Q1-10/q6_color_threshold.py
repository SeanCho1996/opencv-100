import cv2 as cv
from matplotlib import pyplot as plot  # 用pyplot做图
import numpy as np


img = cv.imread('../images/imori.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_thresh = np.zeros_like(img_rgb)
for i in range(0, 3):
    level = img_rgb[:, :, i].astype(np.float)
    level = level // 64 * 64 + 32  # 取模，python求余为%
    img_thresh[:, :, i] = level

plot.figure()
plot.subplot(1, 2, 1)
plot.imshow(img_rgb)
plot.title('rgb')
plot.subplot(1, 2, 2)
plot.imshow(img_thresh)
plot.title('threshed')
plot.show()
