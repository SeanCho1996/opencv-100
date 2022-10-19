# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def translation(img, fx=30, fy=-30):
    M = np.array([[1, 0, fx], [0, 1, fy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# %%
img_t = translation(img_arr)
# %%
