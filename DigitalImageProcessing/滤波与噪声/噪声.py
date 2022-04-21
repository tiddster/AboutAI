import random

from skimage import io
import matplotlib.pyplot as plt
import numpy as np

class Noise():
    def __init__(self, file,SNR=1):
        self.image = io.imread(file)
        self.noiseImg = self.SPNoise(self.image, SNR)

    def setNoise(self):
        imgNoise = self.image
        #上节课已经知道读取的图像是有长、宽和层三个维度
        row, column, layer = imgNoise.shape
        #随机个噪声点
        noiseNum = np.random.randint(2000,5000)
        for i in range(noiseNum):
            #生成一个随机点
            x = np.random.randint(row)
            y = np.random.randint(column)
            # 生成一个随机噪声
            noise = np.random.randint(0,255)
            #将该点对应的值修改为噪声值
            imgNoise[x, y, :] = noise
        return imgNoise

    def SPNoise(self, img, SNR):
        """
        :param img:
        :param SNR: 信噪比
        :return:
        """
        SPimg = img.copy()
        noiseNum = int((1 - SNR) * SPimg.shape[0] * SPimg.shape[1])  #噪声数量
        # 在图像中生成噪声
        for i in range(noiseNum):
            randX = random.randint(0, SPimg.shape[0] - 1)
            randY = random.randint(0, SPimg.shape[1] - 1)
            if random.randint(0, 1) == 0:
                SPimg[randX, randY] = 0
            else:
                SPimg[randX, randY] = 255
        return SPimg

if __name__ == '__main__':
    file = "lena.bmp"
    noise = Noise(file)