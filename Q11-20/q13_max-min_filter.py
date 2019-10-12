import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

h, w = img_gray.shape
filter_size = 3
img_filter = np.zeros_like(img_gray)
for i in range(0+filter_size//2, h-filter_size//2+1):
    for j in range(0 + filter_size // 2, w - filter_size // 2 + 1):
        img_filter[i, j] = np.max(img_gray[i-filter_size//2:i+filter_size//2, j-filter_size//2:j+filter_size//2]) - np.min(img_gray[i-filter_size//2:i+filter_size//2, j-filter_size//2:j+filter_size//2])
        #  当前像素的值等于其8邻域内最大值与最小值的差
plt.figure()
plt.imshow(img_filter, cmap='gray')
plt.show()