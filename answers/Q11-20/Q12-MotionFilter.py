# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% Motion Filter
def motion_filter(img, k_size):
    kernel = np.eye(k_size) / k_size 
    img_fil = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img_fil

# %%
img_fil = motion_filter(img_arr, k_size=3)
# %%
