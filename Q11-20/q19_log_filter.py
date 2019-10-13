import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

h, w = img_gray.shape
sigma = 1.3
filter_log = np.zeros((3, 3), dtype=np.float)

for i in range(0, 3):
    for j in range(0, 3):
        filter_log[i, j] = (i**2 + j**2 - sigma**2) / (2 * np.pi * sigma**6) * np.e**(-(i**2 + j**2)/(2 * sigma**2))
filter_log = filter_log/filter_log.sum()  # 归一化，否则输出灰度值极低

img_filter = cv.filter2D(img_gray, -1, filter_log, borderType=cv.BORDER_CONSTANT)
plt.figure()
plt.imshow(img_filter, cmap='gray')
plt.show()