# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_01_10/imori_noise.jpg")
img_arr = np.array(img)
print(img_arr.shape)
# %% Gaussian Filter
def Gaussian_filter(img, k_size):
    return cv2.GaussianBlur(img, ksize=(k_size, k_size), sigmaX=1.3)
# %%
img_fil = Gaussian_filter(img_arr, 3)