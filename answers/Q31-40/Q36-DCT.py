# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def DCT(img, zip_ratio=0.8, mode="rgb"):
    img_recon = np.zeros_like(img)
    if mode == "rgb":
        for i in range(img.shape[-1]):
            layer = img[:, :, i].astype(np.float32)
            # apply dct
            dct = cv2.dct(layer)
            # compression
            dct_comp = np.zeros_like(dct)
            dct_comp[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)] = dct[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)]
            # retrieve idct
            img_recon[:, :, i] = cv2.idct(dct_comp)
    else:
        dct = cv2.dct(img)
        # compression
        dct_comp = np.zeros_like(dct)
        dct_comp[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)] = dct[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)]
        # retrieve idct
        img_recon = cv2.idct(dct_comp)
    
    return img_recon

# %%
img_com = DCT(img_arr, mode="rgb")

# %%
