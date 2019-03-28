import cv2 as cv
import numpy as np
import math


SRC1 = "./images/Barbara Palvin.jpg"
SRC2 = "./images/Constance Jablonski.jpg"
OUTPUT_GAUSSIAN = "./result/result_gaussian.jpg"
OUTPUT_FFT = "./result/result_fft.jpg"
RADIUS = 10
SIGMA_GAUSSIAN = 10
SIGMA_FFT = 0.985


class MAIN():

    #图片加载
    def __init__(self,src1,src2):
        self.src1 = src1
        self.src2 = src2

    #将两张图片变换为大小一致
    def loadImage(self):
        self.img1 = cv.imread(self.src1)
        self.img2 = cv.imread(self.src2)
        x = self.img1.shape[0]
        y = self.img1.shape[1]
        self.img2 = cv.resize(self.img2,(y,x),interpolation = cv.INTER_CUBIC)

    def get_cv(self,r, sigma):
        return 1 / (2 * math.pi * sigma ** 2) * math.exp((-r ** 2) / (2 * sigma ** 2))




    def getFilter(self,radius,sigma):
        window = np.zeros((radius * 2 + 1, radius * 2 + 1))
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                r = (i ** 2 + j ** 2) ** 0.5
                window[i + radius][j + radius] = self.get_cv(r, sigma)
        return window / np.sum(window)

    def myGaussian(self,src):
        myFilter = self.getFilter(RADIUS,SIGMA_GAUSSIAN)
        print(myFilter)
        # trans = np.zeros(src.shape,np.uint8)
        print(src)
        trans = cv.filter2D(src,3,myFilter)
        print(trans)
        return trans
        # cv.imwrite("result.jpg",trans)
        # print(trans)
        # cv.imshow("result",trans)
        # cv.waitKey()

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
    def hybridImage(self):
        #Gaussian method
        trans1 = self.myGaussian(self.img1)
        trans2 = self.img2 - self.myGaussian(self.img2)
        result = trans2 + trans1
        cv.imwrite(OUTPUT_GAUSSIAN,result)
        #FFT method
        trans1 = self.myFFT(self.img1)
        trans2 = self.myFFT(self.img2)
        h = trans1.shape[0]
        w = trans1.shape[1]

        #中心点坐标
        center_w = w // 2
        center_h = h // 2

        #滤波操作
        for i in range(int(center_h - SIGMA_FFT / 2 * h),int(center_h + SIGMA_FFT / 2 * h)):
            for j in range(int(center_w - SIGMA_FFT / 2 * w),int(center_w + SIGMA_FFT / 2 * w)):
                trans1[i][j] = [0,0,0]

        for i in range(len(trans2)):
            for j in range(len(trans2[0])):
                if trans1[i][j][:].all() != 0:
                    trans2[i][j] = [0,0,0]

        #反傅里叶变换
        img_back = self.myIFFT(trans1 + trans2)
        print(img_back)
        cv.imwrite(OUTPUT_FFT,img_back)






if __name__ == '__main__':
    t = MAIN(SRC1,SRC2)
    t.loadImage()
    t.hybridImage()

