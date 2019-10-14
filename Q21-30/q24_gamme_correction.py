import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('../images/imori_gamma.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

c = 1
g = 2.2
img_out = ((1/c * (img_rgb/255)) ** (1/g))  # 这里输出为[0..1]

# 恢复到0-255
img_out = (img_out * 255).astype(np.uint8)  # 其实不回复到255也可以输出，但是注意输出[0..255]时一定要uint8

plt.figure()
plt.imshow(img_out)
plt.show()