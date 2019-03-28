import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



SRC1 = "./images/Barbara Palvin.jpg"
SRC2 = "./images/Lily Donaldson.jpg"



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



    #自定义傅里叶变换,src为待操作图像
    def myFFT(self,src):
        #trans1盛放傅里叶变换后的结果

        trans = np.zeros(src.shape,np.complex128)
        # print(self.img1[:,:,0])
        for i in range(3):
            f = np.fft.fft2(src[:,:,i])
            # print(f)
            trans[:,:,i] = f[:,:]
            # print(trans[:,:,i])
        return trans
        # print(trans[:,:,i])
        # print(np.abs(img_back))



    def myIFFT(self,src):
        # print("a",src)
        return_val = np.zeros(src.shape,np.uint8)
        for i in range(3):
            f = np.fft.ifft2(src[:,:,i])
            # print(f)
            return_val[:,:,i] = np.abs(f[:,:])
            # print(src[:,:,i])

        return return_val

    def hybridImage(self,sigma):
        # print(self.img1)
        trans1 = self.myFFT(self.img1)
        trans2 = self.myFFT(self.img2)
        h = trans1.shape[0]
        w = trans1.shape[1]
        center_w = w // 2
        center_h = h // 2
        result = trans1[:,:,:]
        for i in range(int(sigma * (center_h - h)),int(sigma * (center_h + h))):
            for j in range(int(sigma * (center_w - w)),int(sigma * (center_w + w))):
                result[i][j] = [0,0,0]
                # print(result[i][j])
                # print(i,j)

        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i][j][:].all() == 0:
                    result[i][j][:] = trans2[i][j][:]
                # else:
                #     result[i][j][:] = 0


        print(trans1)
        print(result)
        img_back = self.myIFFT(trans1)
        # print(img_back)
        cv.imshow("img_back",img_back)
        cv.waitKey()




if __name__ == '__main__':
    t = MAIN(SRC1,SRC2)
    t.loadImage()
    t.hybridImage(0.02)
    # t.myFFT(t.img1,80e3,1)

