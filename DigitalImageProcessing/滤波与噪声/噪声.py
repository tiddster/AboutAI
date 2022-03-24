from skimage import io
import matplotlib.pyplot as plt
import numpy as np

class Noise():
    def __init__(self, file):
        self.image = io.imread(file)

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

if __name__ == '__main__':
    file = "lena.bmp"
    noise = Noise(file)

    io.imshow(noise.setNoise())
    plt.show()