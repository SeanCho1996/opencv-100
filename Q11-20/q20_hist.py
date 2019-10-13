import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_dark.jpg')

plt.figure()
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))  # ravel把原图展成1维， bins就是把0-255分成255份，rwidth对应每一条的宽度，range对应0-255的区间，否则只绘制有直方图分布的区间
plt.show()