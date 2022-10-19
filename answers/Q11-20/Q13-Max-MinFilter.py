# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %% Max-Min
def max_min_filter(img, K_size=3):
    H, W = img.shape

    # Zero padding
    pad = K_size // 2
    out = np.pad(img, ((pad, pad), (pad, pad)), mode="constant")
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.max(tmp[y: y + K_size, x: x + K_size]) - \
                np.min(tmp[y: y + K_size, x: x + K_size])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

# %%
img_dif = max_min_filter(img_gs, 3)
# %%
