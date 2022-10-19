# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img1 = np.array(Image.open("../../ImageProcessing100Wen/Question_51_60/imori.jpg"))
img2 = np.array(Image.open("../../ImageProcessing100Wen/Question_51_60/thorino.jpg"))

# %%
def alpha_blend(img1, img2, alpha=0.5):
    target_shape = img1.shape[0], img1.shape[1]
    img2 = cv2.resize(img2, target_shape)

    return (img1 * alpha + img2 * (1 - alpha)).astype(np.uint8)

# %%
res = alpha_blend(img1, img2)
# %%
