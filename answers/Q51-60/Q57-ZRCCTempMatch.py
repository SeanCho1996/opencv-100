# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
img_part = np.array(Image.open("../../ImageProcessing100Wen/Question_51_60/imori_part.jpg"))
print(f"ori image shape: {img_arr.shape}")
print(f"template shape: {img_part.shape}")
# %%
def ZNCC_template_matching(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_max = cv2.minMaxLoc(res)
    top_left = min_max[3]
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return (top_left, bottom_right)

# %%
tp, br = ZNCC_template_matching(img_arr, img_part)
# %% draw
img_disp = img_arr.copy()
draw = cv2.rectangle(img_disp, tp, br, (0,255,0), 2, 8, 0 )
# %%
