import skimage.io as io
import numpy as np
import math as math
import matplotlib.pyplot as plt


class BilinearI:
    def __init__(self, imageFile='lena.bmp', k=1):
        self.image = io.imread(imageFile)
        #self.layer = self.image[:,:,0]

        self.imageHeight, self.imageWidth, self.imageLayer = np.shape(self.image)
        self.targetImage = np.zeros([self.imageHeight * k, self.imageWidth * k, self.imageLayer],dtype=np.uint8)

        print(self.targetImage.shape)

        # 增大倍数
        self.k = k

    def srcXY(self, dstX, dstY):
        srcX = (dstX + 0.5) / self.k - 1
        srcY = (dstY + 0.5) / self.k - 1
        return srcX, srcY

    def BilinearInterpolation(self):
        targetH,targetW, targetL = self.targetImage.shape
        # 将放大后的坐标映射到原始坐标范围
        H = np.linspace(0, self.imageHeight - 1, targetH - 1)
        W = np.linspace(0, self.imageWidth - 1, targetW - 1)

        for i in range(targetH - 2):
            for j in range(targetW - 2):
                for k in range(targetL):
                    #左上角坐标
                    srcY, srcX = H[i], W[j]
                    #向下取整的左上角坐标
                    srcFloorY, srcFloorX = math.floor(H[i]), math.floor(W[j])
                    a = srcY - srcFloorY
                    b = srcX - srcFloorX
                    # 获得四个点的像素值
                    LT = self.image[srcFloorY][srcFloorX][k]
                    RT = self.image[srcFloorY][srcFloorX + 1][k]
                    LB = self.image[srcFloorY + 1][srcFloorX][k]
                    RB = self.image[srcFloorY + 1][srcFloorX + 1][k]
                    # 使用公式计算映射点像素值
                    self.targetImage[i][j][k] = a * b * LT \
                                       + a * (1-b) * RT \
                                       + (1-a) * b * LB \
                                       + (1-a) * (1-b) * RB
        return self.targetImage

if __name__ == '__main__':
    bi = BilinearI('biu.jpg',k=4)
    bi.BilinearInterpolation()
    io.imshow(bi.BilinearInterpolation())
    plt.show()