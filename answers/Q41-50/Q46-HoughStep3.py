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
def Hough3(img, LT=30, HT=100):
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
       
    def inverse_hough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180 - rho_max / 2

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0, 0, 255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0, 0, 255]
                
        return out.astype(np.uint8)


	# voting
    out = voting(border)
    out = non_maximum_suppression(out)
    out = inverse_hough(out, img_arr)

    return out

# %%
i = Hough3(img_gs)
# %% standard hough
border = cv2.Canny(img_gs, 30, 100)
lines = cv2.HoughLinesP(border, 1, np.pi/180, 105, minLineLength=120, maxLineGap=5)
# %% plot lines
lines1 = lines[:,0,:]#提取为二维
img_plt = img_arr.copy()
for x1,y1,x2,y2 in lines1[:]: 
    cv2.line(img_plt,(x1,y1),(x2,y2),(255,0,0),1)
# %%
