import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



SRC1 = "./images/cat.png"
SRC2 = "./images/people.png"



class MAIN():
    def __init__(self,src1,src2):
        self.src1 = src1
        self.src2 = src2


    def loadImage(self):
        self.img1 = cv.imread(self.src1)
        self.img2 = cv.imread(self.src2)
        x = self.img1.shape[0]
        y = self.img1.shape[1]
        self.img2 = cv.resize(self.img2,(y,x),interpolation = cv.INTER_CUBIC)


    def myFilter(self,sigma):
        kernel_5x5 = np.array([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ])



    #自定义傅里叶变换,src为待操作图像,type决定是高通滤波还是低通滤波
    def myFFT(self,src):
        #trans1盛放傅里叶变换后的结果

        trans = np.zeros(src.shape,np.complex128)
        # print(self.img1[:,:,0])
        for i in range(3):
            f = np.fft.fft2(self.img1[:,:,i])
            # print(f)
            trans[:,:,i] = f[:,:]
            # print(trans[:,:,i])
        return trans
        # print(trans[:,:,i])
        # print(np.abs(img_back))



    def myIFFT(self,src):
        # print("a",src)
        return_val = np.zeros(src.shape)
        for i in range(3):
            f = np.fft.ifft2(src[:,:,i])
            # print(f)
            return_val[:,:,i] = np.abs(f[:,:])
            # print(src[:,:,i])

        return return_val

    def hybridImage(self):
        print(self.img1)
        trans = self.myFFT(self.img1)
        img_back = self.myIFFT(trans)
        print(img_back)





if __name__ == '__main__':
    t = MAIN(SRC1,SRC2)
    t.loadImage()
    t.hybridImage()
    # t.myFFT(t.img1,80e3,1)

