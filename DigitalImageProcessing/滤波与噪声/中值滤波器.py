import numpy as np
from 噪声 import Noise
from skimage import io
import matplotlib.pyplot as plt

class MedianFilter:
    def __init__(self, size):
        self.size = size

    def medianFilter(self, noiseImg):
        fixImg = np.zeros_like(noiseImg)
        r, c = noiseImg.shape[0], noiseImg.shape[1]
        middle = int(self.size / 2)
        for i in range(middle, r-middle):
            for j in range(middle, c-middle):
                for k in range(0,3):
                    fixImg[i][j][k] = np.median(noiseImg[i - middle:i+middle+1, j - middle:j + middle + 1, k])
        return fixImg

    '''
    def getMedian(self,i,j,noiseImg,mid):
        count = 0
        List = np.zeros(self.size * self.size)
        for m in range(i - mid, i + mid + 1):
            for n in range(j - mid, j + mid + 1):
                List[count] = noiseImg[m, n]
                count += 1
        List.sort()
        return List[self.size * self.size // 2]
    '''

if __name__ == '__main__':
    noise1 = Noise('lena.bmp',0.6)
    noise2 = Noise('biu.jpg',0.8)
    ax = plt.subplot(2,3,1)
    ax.set_title('lena_SRC')
    io.imshow(noise1.image)
    ax = plt.subplot(2,3,4)
    ax.set_title('BIUBIUBIU_SRC')
    io.imshow(noise2.image)

    ax = plt.subplot(2,3,2)
    ax.set_title('SNR = 0.8')
    io.imshow(noise1.noiseImg)
    ax = plt.subplot(2,3,5)
    ax.set_title('SNR = 0.6')
    io.imshow(noise2.noiseImg)

    mf = MedianFilter(3)
    fixImg1 = mf.medianFilter(noise1.noiseImg)
    fixImg1 = mf.medianFilter(fixImg1)
    fixImg1 = mf.medianFilter(fixImg1)
    plt.subplot(2,3,3)
    io.imshow(fixImg1)

    fixImg2 = mf.medianFilter(noise2.noiseImg)
    fixImg2 = mf.medianFilter(fixImg2)
    plt.subplot(2,3,6)
    io.imshow(fixImg2)
    plt.show()
