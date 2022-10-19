# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/assets/imori.jpg")
img_arr = np.array(img)
print(img_arr.shape)

# %% 
def color_quantification(img_arr, n_color, bias_value):
    # n_color = 4  # 量化后的颜色块数
    # color_value = 32  # 量化后的偏离值
    return np.array(img_arr // (256 / n_color) * (256 / n_color) + bias_value, dtype=np.uint8)
# %%
