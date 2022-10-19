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
def SAD_template_matching(img, template):
    top_left = -1, -1
    H, W, C = img.shape
    Ht, Wt = template.shape[:-1]
    v = 255 * H * W * C
    for y in range(H-Ht):
        for x in range(W-Wt):
            _v = np.sum(np.abs(img[y:y+Ht, x:x+Wt] - template))
            if _v < v:
                v = _v
                top_left = x, y
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return (top_left, bottom_right)

# %%
tp, br = SAD_template_matching(img_arr, img_part)
# %% draw
img_disp = img_arr.copy()
draw = cv2.rectangle(img_disp, tp, br, (0,255,0), 2, 8, 0 )

# %%
