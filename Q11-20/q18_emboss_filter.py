import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

filter_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

img_filter = cv.filter2D(img_gray, -1, filter_emboss, borderType=cv.BORDER_CONSTANT)

plt.figure()
plt.imshow(img_filter, cmap='gray')
plt.show()