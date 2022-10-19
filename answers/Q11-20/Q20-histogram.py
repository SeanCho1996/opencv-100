# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_11_20/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %% 
def hist(img):
    return cv2.calcHist([img], [0], None, [256], [0,256])

# %%
h = hist(img_gs)
# %%
