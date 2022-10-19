# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
thresh, bin = cv2.threshold(img_gs, None, maxval=255, type=cv2.THRESH_OTSU)

# %%
def close_bin(img, se_form="ellipse", k_size=(3, 3)):
    se_dict = {
        "ellipse": cv2.MORPH_ELLIPSE, 
        "cross": cv2.MORPH_CROSS,
        "rect": cv2.MORPH_RECT
    }
    se = cv2.getStructuringElement(se_dict[se_form], k_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

# %%
res = close_bin(bin, se_form="cross")
# %%
