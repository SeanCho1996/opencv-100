import cv2 as cv
from matplotlib import pyplot as plot


img = cv.imread('../images/imori.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plot.figure()
plot.subplot(1, 2, 1)
plot.imshow(img)
plot.subplot(1, 2, 2)
plot.imshow(img_gray, cmap='gray')  # 此处需要添加colormap选项，否则图片虽然已经转换为灰度，但显示为绿色
plot.show()
