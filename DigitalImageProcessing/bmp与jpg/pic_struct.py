from array import array
from struct import pack, unpack

import numpy as np


class PIC:
    def __init__(self, file_path=None):
        self.color_type = False                 #单色图像或是彩色图像
        self.M = b"0"                           #保留字段
        self.col = 512                          #图像列数
        self.row = 512                          #图像行数
        self.col_start = 0                      #图像列起点
        self.row_start = 0                      #图像行起点
        self.N = 0                              #保留字段
        self.others = b""                       #其他
        self.notes = b""                        #注释信息
        self.notes_size = len(self.notes)       #注释区字节数

        self.image = np.array([])

    # 将bmp信息储存进pic文件中
    def bmp2pic(self, bmp):
        self.col = bmp.width
        self.row = bmp.height

        self.image = bmp.image

        if bmp.image.shape[2] == 3:
            self.color_type = True
        else:
            self.color_type = False

    # 储存pic文件
    def save_pic(self,pic_path):
        f = open(pic_path, "wb")
        data = pack(f"?chhhhhh50s{self.notes_size}s", self.color_type, self.M, self.notes_size,
                    self.col, self.row, self.col_start, self.row_start, self.N, self.others, self.notes)
        f.write(data)
        rgb = self.image.ravel()
        for c in self.image:
            f.write(c)
        f.close()

    # 读取pic格式的头文件信息以及rgb信息
    def load_pic(self, pic_path):
        f = open(pic_path, "rb")
        data = f.read(64)
        data = unpack(f"?chhhhhh50s{self.notes_size}s", data)
        self.pic_info(data)
        rgb_data = f.read()
        rgb_list = array('B', rgb_data)
        rgb_mat = np.reshape(rgb_list, (self.row, self.col, 3))

        return data, rgb_mat

    # 将读取的信息存入pic结构中
    def pic_info(self, data):
        self.color_type = data[0]
        self.M = data[1]
        self.notes_size = data[2]
        self.col = data[3]
        self.row = data[4]
        self.col_start = data[5]
        self.row_start = data[6]
        self.N = data[7]
        self.others = data[8]
        self.notes = data[9]
