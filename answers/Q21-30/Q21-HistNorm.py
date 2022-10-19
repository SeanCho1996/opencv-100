# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori_dark.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def hist_norm(img, interval=(0, 255)):
    print(f"original color interval: ({np.min(img)}, {np.max(img)})")
    assert len(interval) == 2, "hist interval must be a length 2 tuple"
    assert interval[0] >= 0, "min of target interval must >= 0"
    assert interval[1] <= 255, "max of target interval must <= 255"
    assert interval[0] < interval[1], "interval min must < interval max"

    out = (interval[1] - interval[0]) / (np.max(img) - np.min(img)) * (img - np.min(img)) + interval[0]
    out[out < interval[0]] = interval[0]
    out[out > interval[1]] = interval[1]

    return out.astype(np.uint8)

# %%
img_norm = hist_norm(img_arr)

