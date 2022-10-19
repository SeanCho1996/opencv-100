# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def hist_equalize(img):
    print(f"original color interval: ({np.min(img)}, {np.max(img)})")
    
    n_pix = img.size
    sum_h = 0.
    out = img.copy()

    for i in range(0, 256):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = np.max(img) / n_pix * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)
    return out

# %%
img_norm = hist_equalize(img_arr)


# %%
