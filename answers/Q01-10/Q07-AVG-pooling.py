# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% AVG_pooling
def avg_pool(img, k_size):
    kernel = np.ones((k_size, k_size)) / (k_size ** 2)
    img_fil = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    mesh = np.meshgrid([i * k_size for i in range(int(img.shape[0] // k_size))], [i * k_size for i in range(int(img.shape[1] // k_size))])
    return np.transpose(img_fil[tuple(mesh)], (1, 0, -1))
# %%
img_pool = avg_pool(img_arr, 8)

