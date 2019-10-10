import cv2 as cv
from matplotlib import pyplot as plot  # 用pyplot做图


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, img_bw = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY)

plot.figure()
plot.imshow(img_bw, cmap='gray')
plot.show()
