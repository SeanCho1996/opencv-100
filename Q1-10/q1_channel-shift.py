import cv2 as cv
from matplotlib import pyplot as plot  # 用pyplot做图


img = cv.imread('../images/imori.jpg')  # opencv读取图片实际上是BGR通道
img_c = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plot.figure()
plot.subplot(1, 2, 1)
plot.imshow(img)
plot.subplot(1, 2, 2)
plot.imshow(img_c)
plot.show()
