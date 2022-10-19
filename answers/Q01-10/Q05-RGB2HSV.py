# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
img_hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
# %%
