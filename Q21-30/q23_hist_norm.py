import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('../images/imori_dark.jpg').astype(np.uint8)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

c = np.min(img_rgb)
d = np.max(img_rgb)

img_norm = (255 / (d - c) * (img_rgb - c)).astype(np.uint8)

plt.figure()
plt.imshow(img_norm)
plt.show()
