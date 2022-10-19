# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def rotation(img, angle=30):
    M = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[0] // 2), angle, 1)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# %%
img_t = rotation(img_arr)
# %%
