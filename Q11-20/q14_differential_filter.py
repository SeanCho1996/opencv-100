import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

diff_h = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
diff_v = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])

img_h = cv.filter2D(img_gray, -1, diff_h, borderType=cv.BORDER_CONSTANT)
img_v = cv.filter2D(img_gray, -1, diff_v, borderType=cv.BORDER_CONSTANT)  # 由于灰度图为单层，所以ddepth参数取-1令其与输入图像层数一致

plt.figure()
plt.subplot(1, 3, 1)
plt.title('original')
plt.imshow(img_gray, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('horizon')
plt.imshow(img_h, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('vertical')
plt.imshow(img_v, cmap='gray')
plt.show()