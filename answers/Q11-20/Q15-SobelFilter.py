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
def SobelFilter(img, k_size=3):
    return cv2.Sobel(img, -1, 1, 0, ksize=k_size), cv2.Sobel(img, -1, 0, 1, ksize=k_size)

# %%
diff_x, diff_y = SobelFilter(img_gs)
# %%
