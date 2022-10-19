# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %%
def laplacian_filter(img, k_size=3):
    return cv2.Laplacian(img, -1, ksize=k_size)

# %%
diff = laplacian_filter(img_gs)
# %%
