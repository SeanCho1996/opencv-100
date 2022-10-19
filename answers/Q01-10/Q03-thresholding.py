# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% thresholding
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

_, img_bin = cv2.threshold(img_gs, 128, maxval=255, type=cv2.THRESH_BINARY)
# %%
