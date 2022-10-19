# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_01_10/imori_noise.jpg")
img_arr = np.array(img)
print(img_arr.shape)
# %% Median Filter
def median_filter(img, k_size):
    return cv2.medianBlur(img, ksize=k_size)
# %%
img_fil = median_filter(img_arr, 3)
# %%
