# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %%
def canny1(img, k_size=5, std=1.4):
    img_gauss = cv2.GaussianBlur(img, ksize=(k_size, k_size), sigmaX=std)
    sobel_x = cv2.Sobel(img_gauss, -1, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gauss, -1, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    arctan = np.arctan(sobel_y / np.maximum(sobel_x, 1e-5))
    angle = arctan / np.pi * 180
    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype=np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

    return _angle.astype(np.uint8)

# %%
grad = canny1(img_gs)
# %%
