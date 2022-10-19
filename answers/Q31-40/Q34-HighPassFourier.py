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
def high_pass_transform(img, mask_ratio=(0.2, 0.2), mode="rgb"):
    # generate high pass filter
    mask_size = [int(img.shape[i] * mask_ratio[i]) for i in range(2)]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mask_size)
    pad_size = ((int((img.shape[0] - mask_size[0]) // 2),
                img.shape[0] - int((img.shape[0] - mask_size[0]) // 2) - mask_size[0]),
                (int((img.shape[1] - mask_size[1]) // 2),
                img.shape[1] - int((img.shape[1] - mask_size[1]) // 2) - mask_size[1]))
    mask = 1 - np.pad(se, pad_size)
    
    if mode == "rgb":
        img_inv = np.zeros_like(img)
        for i in range(3):
            # fft
            fft = np.fft.fft2(img[:, :, i])
            fft_shift = np.fft.fftshift(fft)        
    
            # filter operation
            fft_high = fft_shift * mask

            # ifft
            ishift = np.fft.ifftshift(fft_high)  #将零频率分量还原      (ishift == f).sum() = 512*512
            iimg = np.fft.ifft2(ishift)  #逆傅里叶变换,变换后的结果还是一个复数数组
            img_inv[:, :, i] = np.abs(iimg)    #将复数数组转换到[0,256]灰度区间内
    else:
        # fft
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)        

        # filter operation
        fft_low = fft_shift * mask

        # ifft
        ishift = np.fft.ifftshift(fft_low)  #将零频率分量还原      (ishift == f).sum() = 512*512
        iimg = np.fft.ifft2(ishift)  #逆傅里叶变换,变换后的结果还是一个复数数组
        img_inv = np.abs(iimg)    #将复数数组转换到[0,256]灰度区间内

    return img_inv

# %%
i = high_pass_transform(img_gs, mode="gs")
# %%
