# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def scaling(img, scale=(1.3, 0.8), method="bilinear"):
    d_size = (int(img.shape[1] * scale[0]), int(img.shape[0] * scale[1]))
    dict_method = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC
    }
    return cv2.resize(img, d_size, interpolation=dict_method[method])

# %%
img_a = scaling(img_arr)
# %%
