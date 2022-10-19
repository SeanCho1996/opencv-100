# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% MAX_pooling
def max_pool(img, k_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    img_dil = cv2.dilate(img, kernel)
    mesh = np.meshgrid([i * k_size for i in range(int(img.shape[0] // k_size))], [i * k_size for i in range(int(img.shape[1] // k_size))])
    return np.transpose(img_dil[tuple(mesh)], (1, 0, -1))

# %%
img_max = max_pool(img_arr, 8)
# %%
