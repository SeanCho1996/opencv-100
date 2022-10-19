# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %%
def toYCrCb(img, ch=0, ratio=0.7):
    assert ch < 3, "channel must inside RGB"
    img_t = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    img_t[:, :, ch] *= ratio
    img_t[img_t > 255] = 255
    img_m = cv2.cvtColor(img_t.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return img_m

# %%
img_m = toYCrCb(img_arr)
# %%
