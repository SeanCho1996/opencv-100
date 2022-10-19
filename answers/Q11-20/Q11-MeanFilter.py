# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% Mean filter
def mean_filter(img, k_size):
    kernel = np.ones((k_size, k_size)) / (k_size ** 2)
    img_fil = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img_fil
# %%
img_fil = mean_filter(img_arr, 3)
# %%
