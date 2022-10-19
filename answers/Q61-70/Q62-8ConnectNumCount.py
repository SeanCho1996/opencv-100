# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = np.array(Image.open("../../ImageProcessing100Wen/Question_61_70/renketsu.png")).astype(np.float32)

# %%
def connect_count8(img):
    assert len(img.shape) == 2, "input image must be single channel"
    # get shape
    H, W = img.shape
    img = np.expand_dims(img, -1)
    img = np.concatenate((img, img, img), -1)

    # prepare temporary
    _tmp = np.zeros((H, W), dtype=np.int)

    # get binarize
    _tmp[img[..., 0] > 0] = 1

    # inverse for connect 8
    tmp = 1 - _tmp

    # prepare image
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # each pixel
    for y in range(H):
        for x in range(W):
            if _tmp[y, x] < 1:
                continue

            S = 0
            S += (tmp[y,min(x+1,W-1)] - tmp[y,min(x+1,W-1)] * tmp[max(y-1,0),min(x+1,W-1)] * tmp[max(y-1,0),x])
            S += (tmp[max(y-1,0),x] - tmp[max(y-1,0),x] * tmp[max(y-1,0),max(x-1,0)] * tmp[y,max(x-1,0)])
            S += (tmp[y,max(x-1,0)] - tmp[y,max(x-1,0)] * tmp[min(y+1,H-1),max(x-1,0)] * tmp[min(y+1,H-1),x])
            S += (tmp[min(y+1,H-1),x] - tmp[min(y+1,H-1),x] * tmp[min(y+1,H-1),min(x+1,W-1)] * tmp[y,min(x+1,W-1)])
            
            if S == 0:
                out[y,x] = [0, 0, 255]
            elif S == 1:
                out[y,x] = [0, 255, 0]
            elif S == 2:
                out[y,x] = [255, 0, 0]
            elif S == 3:
                out[y,x] = [255, 255, 0]
            elif S == 4:
                out[y,x] = [255, 0, 255]
                    
    out = out.astype(np.uint8)

    return out

# %%
out = connect_count8(img)

# %%
