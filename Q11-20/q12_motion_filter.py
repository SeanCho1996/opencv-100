import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori_noise.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

motion_filter = 1/3 * np.eye(3, 3)
img_filter = cv.filter2D(img_rgb, img_rgb.shape[2], motion_filter, borderType=cv.BORDER_CONSTANT)

plt.figure()
plt.imshow(img_filter)
plt.show()