import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



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
        #逐通道进行傅里叶变换
        trans = np.zeros(src.shape,np.complex128)
        for i in range(3):
            f = np.fft.fft2(src[:,:,i])
            trans[:,:,i] = f[:,:]
        return trans



    def myIFFT(self,src):
        # 务必用uint16否则出现噪点
        return_val = np.zeros(src.shape,np.uint16)
        #逐通道反傅里叶变换
        for i in range(3):
            f = np.fft.ifft2(src[:,:,i])
            return_val[:,:,i] = np.abs(f[:,:])

        return return_val

    #图像混合操作
    def hybridImage(self,sigma):
        trans1 = self.myFFT(self.img1)
        trans2 = self.myFFT(self.img2)
        h = trans1.shape[0]
        w = trans1.shape[1]

        #中心点坐标
        center_w = w // 2
        center_h = h // 2

        #滤波操作
        for i in range(int(center_h - sigma / 2 * h),int(center_h + sigma / 2 * h)):
            for j in range(int(center_w - sigma / 2 * w),int(center_w + sigma / 2 * w)):
                trans1[i][j] = [0,0,0]

        for i in range(len(trans2)):
            for j in range(len(trans2[0])):
                if trans1[i][j][:].all() != 0:
                    trans2[i][j] = [0,0,0]
        img_back = self.myIFFT(trans1 + trans2)
        print(img_back)
        cv.imwrite("result.jpg",img_back)






if __name__ == '__main__':
    t = MAIN(SRC1,SRC2)
    t.loadImage()
    t.hybridImage(0.99)

