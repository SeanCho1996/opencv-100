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
def Hough2(img, LT=30, HT=100):
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
    
    def non_maximum_suppression(hough):
       rho_max, _ = hough.shape

       ## non maximum suppression
       for y in range(rho_max):
           for x in range(180):
               # get 8 nearest neighbor
               x1 = max(x-1, 0)
               x2 = min(x+2, 180)
               y1 = max(y-1, 0)
               y2 = min(y+2, rho_max-1)
               if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
                   pass
                   #hough[y,x] = 255
               else:
                   hough[y,x] = 0

       # for hough visualization
       # get top-10 x index of hough table
       ind_x = np.argsort(hough.ravel())[::-1][:20]
       # get y index
       ind_y = ind_x.copy()
       thetas = ind_x % 180
       rhos = ind_y // 180
       _hough = np.zeros_like(hough, dtype=np.int)
       _hough[rhos, thetas] = 255

       return _hough

	# voting
    out = voting(border)
    out = non_maximum_suppression(out)

    return out

# %%
i = Hough2(img_gs)
# %%
