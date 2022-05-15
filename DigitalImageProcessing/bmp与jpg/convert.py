import cv2
from matplotlib import pyplot as plt

from DigitalImageProcessing.bmp与jpg.bmp_struct import BMP
from DigitalImageProcessing.bmp与jpg.pic_struct import PIC


# bmp转pic函数
def bmp2pic(bmp, pic):
    pic.bmp2pic(bmp)
    pic.save_pic('lena.pic')

# pic转bmp函数
def pic2bmp(pic, bmp):
    bmp.pic2bmp(pic)
    bmp.save_bmp('lena_copy.bmp')

if __name__ == '__main__':
    bmp = BMP('lena.bmp')
    pic = PIC()
    bmp2pic(bmp,pic)
    picImage = pic.load_pic('lena.pic')[1]
    tempImage = picImage.copy()
    cv2.flip(picImage, 0, tempImage)
    plt.imshow(tempImage)
    plt.show()

    pic2bmp(pic,bmp)
    bmpImage = bmp.load_bmp_rgb('lena_copy.bmp')
    tempImage = bmpImage.copy()
    cv2.flip(picImage, 0, tempImage)
    plt.imshow(tempImage)
    plt.show()