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
def differential_filter(img):
    k_x = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.int)
    k_y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int)

    img_diff_x = cv2.filter2D(img, -1, k_x, borderType=cv2.BORDER_CONSTANT)
    img_diff_y = cv2.filter2D(img, -1, k_y, borderType=cv2.BORDER_CONSTANT)
    return img_diff_x, img_diff_y

# %%
diff_x, diff_y = differential_filter(img_gs)
# %%
