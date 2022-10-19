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

def PSNR(img1, img2):
    diff = cv2.absdiff(img1, img2).astype(np.float32)
    diff *= diff
    sse = np.sum(diff)
    mse = sse / img1.size
    return 10.0 * np.log10((255 * 255) / mse)

# %%
img_comp = DCT(img_arr)
print(f"PSNR: {PSNR(img_arr, img_comp)}")
# %%
