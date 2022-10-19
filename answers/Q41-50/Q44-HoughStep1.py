# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_41_50/thorino.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %%
def Hough1(img, LT=30, HT=100):
    border = cv2.Canny(img, LT, HT)

	## Voting
    def voting(edge):
       H, W = edge.shape
       drho = 1
       dtheta = 1
    
       # get rho max length
       rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)
    
       # hough table
       hough = np.zeros((rho_max * 2, 180), dtype=np.int)
    
       # get index of edge
       ind = np.where(edge == 255)
    
       ## hough transformation
       for y, x in zip(ind[0], ind[1]):
               for theta in range(0, 180, dtheta):
                       # get polar coordinat4s
                       t = np.pi / 180 * theta
                       rho = int(x * np.cos(t) + y * np.sin(t))
    
                       # vote
                       hough[rho + rho_max, theta] += 1
                            
       out = hough.astype(np.uint8)
    
       return out

	# voting
    out = voting(border)

    return out

# %%
i = Hough1(img_gs)
# %%
