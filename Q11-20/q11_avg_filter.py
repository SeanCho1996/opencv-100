import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_noise.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

avg_filter = 1/9 * np.ones((3, 3))
img_filter = cv.filter2D(img_rgb, img_rgb.shape[2], avg_filter, borderType=cv.BORDER_CONSTANT)

plt.figure()
plt.imshow(img_filter)
plt.show()
