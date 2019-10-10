import cv2 as cv
from matplotlib import pyplot as plot  # 用pyplot做图
import numpy as np


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

x = img.shape[0]
y = img.shape[1]
th_final = 0
sb2_final = 0

# 从图中最小灰度值处去到最大灰度值，避免空矩阵出现
img_min = np.min(img_gray)
img_max = np.max(img_gray)


for th in range(img_min, img_max):
    v0 = img_gray[np.where(img_gray > th)]  # 用numpy的函数可以避免两次循环，减少运行时间
    m0 = np.mean(v0)
    w0 = len(v0)/(x*y)
    v1 = img_gray[np.where(img_gray <= th)]
    m1 = np.mean(v1)
    w1 = len(v1)/(x*y)
    sb2 = w1 * w0 * ((m1-m0)**2)  # 寻找最大类间方差时的阈值作为最终二值化阈值，这是大津二值方法的原理
    if sb2 > sb2_final:
        sb2_final = sb2
        th_final = th

print(th_final)
ret, img_bw1 = cv.threshold(img_gray.astype(np.float), th_final, 255, cv.THRESH_BINARY)
ret, img_bw2 = cv.threshold(img_gray.astype(np.float), 128, 255, cv.THRESH_BINARY)
plot.figure()
plot.subplot(1, 2, 1)
plot.imshow(img_bw1, cmap='gray')
plot.title('threshold otsu')
plot.subplot(1, 2, 2)
plot.imshow(img_bw2, cmap='gray')
plot.title('threshold 128')
plot.show()
