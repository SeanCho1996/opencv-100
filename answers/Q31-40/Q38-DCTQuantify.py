# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def DCTQuantization(img, zip_ratio=0.8, mode="rgb"):
    T = 8
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                  (12, 12, 14, 19, 26, 58, 60, 55),
                  (14, 13, 16, 24, 40, 57, 69, 56),
                  (14, 17, 22, 29, 51, 87, 80, 62),
                  (18, 22, 37, 56, 68, 109, 103, 77),
                  (24, 35, 55, 64, 81, 104, 113, 92),
                  (49, 64, 78, 87, 103, 121, 120, 101),
                  (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    img_recon = np.zeros_like(img)
    if mode == "rgb":
        for i in range(img.shape[-1]):
            layer = img[:, :, i].astype(np.float32)
            # apply dct
            dct = cv2.dct(layer)
            # compression
            dct_comp = np.zeros_like(dct)
            dct_comp[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)] = dct[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)]
            # quantization
            for ys in range(0, dct.shape[0], T):
                for xs in range(0, dct.shape[1], T):
                        dct_comp[ys: ys + T, xs: xs + T] = np.round(dct_comp[ys: ys + T, xs: xs + T] / Q) * Q
            # retrieve idct
            img_recon[:, :, i] = cv2.idct(dct_comp)
    else:
        dct = cv2.dct(img)
        # compression
        dct_comp = np.zeros_like(dct)
        dct_comp[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)] = dct[0 : int(img.shape[0] * zip_ratio), 0 : int(img.shape[1] * zip_ratio)]
        # quantization
        for ys in range(0, dct.shape[0], T):
                for xs in range(0, dct.shape[1], T):
                        dct_comp[ys: ys + T, xs: xs + T] = np.round(dct_comp[ys: ys + T, xs: xs + T] / Q) * Q
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
img_comp = DCTQuantization(img_arr)
print(f"PSNR: {PSNR(img_arr, img_comp)}")
# %%
