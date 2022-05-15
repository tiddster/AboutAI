from array import array
from struct import unpack, pack

import numpy as np
from matplotlib import pyplot as plt

from DigitalImageProcessing import utils

import cv2


class BMP:
    def __init__(self, file_path):
        tp = self.load_bmp_header(file_path)

        self.tag = tp[0]
        self.fileSize = tp[1]
        self.A = tp[2]
        self.B = tp[3]
        self.rgbOffset = tp[4]
        self.infoSize = tp[5]
        self.width = tp[6]
        self.height = tp[7]
        self.pane = tp[8]
        self.color = tp[9]
        self.compress = tp[10]
        self.rgbSize = tp[11]
        self.C, self.D, self.E, self.F = tp[12:16]

        self.image = self.load_bmp_rgb(file_path)

    def print_bmp_header(self):
        print("tag      :{}".format(self.tag))
        print("fileSize :{}".format(self.fileSize))
        print("rgbOffset:{}".format(self.rgbOffset))
        print("infoSize :{}".format(self.infoSize))
        print("width    :{}".format(self.width))
        print("height   :{}".format(self.height))
        print("pane     :{}".format(self.pane))
        print("color    :{}".format(self.color))
        print("compress :{}".format(self.compress))
        print("rgbSize  :{}".format(self.rgbSize))

    #将pic信息储存在bmp中
    def pic2bmp(self, pic):
        self.tag = b'BM'
        self.rgbOffset = 54
        self.width = pic.col
        self.height = pic.row
        self.rgbSize = self.width * self.height * 3
        self.fileSize = self.rgbSize + self.rgbOffset

        self.image = pic.image

    # 储存bmp信息
    def save_bmp(self, file_path):
        f = open(file_path, 'wb')
        data = pack('<2sI2H4I2H6I', self.tag, self.fileSize, self.A, self.B, self.rgbOffset, self.infoSize,
                    self.width, self.height, self.pane, self.color, self.compress, self.rgbSize, self.C, self.D, self.E, self.F)
        f.write(data)
        rgb = self.image.ravel()
        for c in rgb:
            f.write(c)
        f.close()

    # 读取bmp位图头信息
    def load_bmp_header(self, file_path):
        f = open(file_path, 'rb')
        data = f.read(0x36)
        bmp_info = unpack('<2sI2H4I2H6I', data)
        print(bmp_info)
        return bmp_info

    # 读取bmp图像rgb信息
    def load_bmp_rgb(self, file_path):
        f = open(file_path, 'rb')
        f.read(0x36)

        rgb_data = f.read()
        rgb_list = array('B', rgb_data)
        rgb_mat = np.reshape(rgb_list, (self.height, self.width, 3))

        return rgb_mat


if __name__ == '__main__':
    bmp = BMP(utils.lena_path)
    bmp.print_bmp_header()
    tempImage = bmp.image.copy()
    cv2.flip(bmp.image, 0, tempImage)
    plt.imshow(tempImage)
    plt.show()
