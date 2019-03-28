import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./images/cat.png",0) #直接读为灰度图像
print(img)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到0-255
s1 = np.log(np.abs(fshift))
plt.subplot(131),plt.imshow(img,'gray'),plt.title('original')
plt.subplot(132),plt.imshow(s1,'gray'),plt.title('center')
# 逆变换
# f1shift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f)
#出来的是复数，无法显示
img_back = np.abs(img_back)
print(img_back)
plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('img back')
plt.show()