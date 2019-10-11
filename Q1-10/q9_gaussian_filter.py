import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_noise.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

g_filter1 = cv.getGaussianKernel(3, 1.3)
g_filter2 = cv.getGaussianKernel(3, 1.3)
g_filter = g_filter1 * np.transpose(g_filter2)

img_filter = cv.filter2D(img_rgb, img_rgb.shape[2], g_filter, borderType=cv.BORDER_CONSTANT)  # 边缘选项选择constant似乎直接默认为zeropadding????
plt.figure()
plt.imshow(img_filter)
plt.show()