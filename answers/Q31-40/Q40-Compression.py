# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_31_40/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def compression(img):
    assert len(img.shape) == 3, "channel must inside RGB"
    
    img_recon = np.zeros_like(img)
    T = 8
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                  (12, 12, 14, 19, 26, 58, 60, 55),
                  (14, 13, 16, 24, 40, 57, 69, 56),
                  (14, 17, 22, 29, 51, 87, 80, 62),
                  (18, 22, 37, 56, 68, 109, 103, 77),
                  (24, 35, 55, 64, 81, 104, 113, 92),
                  (49, 64, 78, 87, 103, 121, 120, 101),
                  (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
                  (18, 21, 26, 66, 99, 99, 99, 99),
                  (24, 26, 56, 99, 99, 99, 99, 99),
                  (47, 66, 99, 99, 99, 99, 99, 99),
                  (99, 99, 99, 99, 99, 99, 99, 99),
                  (99, 99, 99, 99, 99, 99, 99, 99),
                  (99, 99, 99, 99, 99, 99, 99, 99),
                  (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

    # RGB to YCrCb
    img_t = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    # DCT and quantization on each channel
    for i in range(img_t.shape[-1]):
        layer = img_t[:, :, i].astype(np.float32)
        # apply dct
        dct = cv2.dct(layer)
        # quantization
        if i == 0:
            for ys in range(0, dct.shape[0], T):
                for xs in range(0, dct.shape[1], T):
                    dct[ys: ys + T, xs: xs + T] = np.round(dct[ys: ys + T, xs: xs + T] / Q1) * Q1
        else:
            for ys in range(0, dct.shape[0], T):
                for xs in range(0, dct.shape[1], T):
                    dct[ys: ys + T, xs: xs + T] = np.round(dct[ys: ys + T, xs: xs + T] / Q2) * Q2
        # IDCT
        img_recon[:, :, i] = cv2.idct(dct)
    img_recon_t = cv2.cvtColor(img_recon.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return img_recon_t
# %%
img_m = compression(img_arr)
# %%
