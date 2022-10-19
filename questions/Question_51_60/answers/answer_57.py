import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

mi = np.mean(img)

# Read templete image
temp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = temp.shape

mt = np.mean(temp)

# Templete matching
i, j = -1, -1
v = -1
for y in range(H-Ht):
    for x in range(W-Wt):
        _v = np.sum((img[y:y+Ht, x:x+Wt]-mi) * (temp-mt))
        _v /= (np.sqrt(np.sum((img[y:y+Ht, x:x+Wt]-mi)**2)) * np.sqrt(np.sum((temp-mt)**2)))
        if _v > v:
            v = _v
            i, j = x, y

out = img.copy()
cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0,0,255), thickness=1)
out = out.astype(np.uint8)
                
# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
