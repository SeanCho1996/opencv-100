# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori_dark.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def hist_gauss(img, mean=128, std=52):
    ori_mean = np.mean(img)
    ori_std = np.std(img)
    print(f"original mean-sigma: ({ori_mean}, {ori_std})")
    
    out = std / ori_std * (img - ori_mean) + mean
    out[out < 0] = 0
    out[out > 255] = 255

    return out.astype(np.uint8)

# %%
img_norm = hist_gauss(img_arr)


# %%
