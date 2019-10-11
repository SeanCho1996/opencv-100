import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_noise.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_filter = cv.medianBlur(img_rgb, 3)  # 由于中值滤波为非线性滤波，所以不需要构造滤波器，直接中值模糊即可
plt.figure()
plt.imshow(img_filter)
plt.show()