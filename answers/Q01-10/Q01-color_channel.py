# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# 为什么先用pil读图像，再转成np.array呢，因为如果直接用cv2.imread，
# 图像会直接变成BGR通道，接下来如果用这个数组做运算会出现问题，用
# PIL再转numpy可以保持RGB

# %% switch channel
img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
