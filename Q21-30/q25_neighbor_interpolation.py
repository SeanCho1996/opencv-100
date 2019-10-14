import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../images/imori.jpg').astype(np.uint8)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

n = 1.5
img_out = np.zeros([(np.floor(img_rgb.shape[0]*1.5)).astype(np.uint8), (np.floor(img_rgb.shape[1]*1.5)).astype(np.uint8), img_rgb.shape[2]])

for c in range(img_out.shape[2]):
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            img_out[i, j, c] = img_rgb[(np.floor(i/n)).astype(np.uint8), (np.floor(j/n)).astype(np.uint8), c]
img_out = img_out.astype(np.uint8)

plt.figure()
plt.imshow(img_out)
plt.show()