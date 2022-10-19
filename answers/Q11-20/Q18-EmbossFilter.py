# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %%
def emboss_filter(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(img, -1, kernel)

# %%
img_fil = emboss_filter(img_gs)
# %%
