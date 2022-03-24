from matplotlib import pyplot as plt
from skimage import io

from DigitalImageProcessing.滤波与噪声.噪声 import Noise
from DigitalImageProcessing.滤波与噪声.高斯滤波 import GaussModel

if __name__ == '__main__':
    gm = GaussModel(5)
    gm.setGaussModel()
    file = "lena.bmp"
    noise = Noise(file)
    noiseImg = noise.setNoise()

    gaussImg = gm.gaussFilter(noiseImg)
    print(gaussImg)

    plt.subplot(1,2,1)
    io.imshow(gaussImg)

    plt.subplot(1,2,2)
    io.imshow(noiseImg)

    plt.show()