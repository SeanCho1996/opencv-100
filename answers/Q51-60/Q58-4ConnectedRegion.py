# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = np.array(Image.open("../../ImageProcessing100Wen/Question_51_60/seg.png"))

# %%
def connected_components4(img):
    num_comp, labels = cv2.connectedComponents(img, connectivity=4)
    return num_comp, labels
# %%
num, l = connected_components4(img) # num is the number of connected components, l is segmented image
# %%
