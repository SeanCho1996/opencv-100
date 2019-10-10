import cv2 as cv
from matplotlib import pyplot as plot  # 用pyplot做图
import numpy as np


img = cv.imread('../images/imori.jpg')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h = img_hsv[:, :, 0]
h_inv = abs(h-180)

img_n_hsv = np.zeros_like(img)
img_n_hsv[:, :, 0] = h_inv
img_n_hsv[:, :, 1] = img_hsv[:, :, 1]
img_n_hsv[:, :, 2] = img_hsv[:, :, 2]
img_rgb = cv.cvtColor(img_n_hsv, cv.COLOR_HSV2RGB)

plot.figure()
plot.imshow(img_rgb)
plot.show()
