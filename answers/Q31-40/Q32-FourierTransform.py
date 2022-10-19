# %% import 
import cv2
from PIL import Image
import numpy as np

# %% load original image
img = Image.open("../../ImageProcessing100Wen/Question_21_30/imori.jpg")
img_arr = np.array(img)
img_gs = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
print(img_gs.shape)

# %%
def Fourier_transform(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    spectrum_view = 20*np.log(np.abs(fft_shift))   #为显示图像，将复数值调到[0,256]的灰度空间内。
    return fft_shift, spectrum_view

# %%
fft, fft_view = Fourier_transform(img_gs)
# %%
