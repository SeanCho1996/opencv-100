# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_11_20/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %% LoG Filter
def LoG_filter(img, k_size=3, sigma=1.3):
    H, W = img.shape

    # Zero padding
    pad = k_size // 2
    out = np.pad(img, ((pad, pad), (pad, pad)), mode="constant")

    filter_log = np.zeros((3, 3), dtype=np.float)
    for i in range(0, k_size):
        for j in range(0, k_size):
            filter_log[i, j] = (i**2 + j**2 - sigma**2) / (2 * np.pi * sigma**6) * np.exp(-(i**2 + j**2)/(2 * sigma**2))
    filter_log /= filter_log.sum()  # 归一化，否则输出灰度值极低
    out = cv2.filter2D(img, -1, filter_log, borderType=cv2.BORDER_CONSTANT)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return np.clip(out, 0, 255)
# %%
img_fil = LoG_filter(img_gs)
# %%
