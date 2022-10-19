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
def PrewittFilter(img):
    k_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    k_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img, -1, k_x)
    img_prewitty = cv2.filter2D(img, -1, k_y)
    return img_prewittx, img_prewitty
# %%
diff_x, diff_y = PrewittFilter(img_gs)
# %%
