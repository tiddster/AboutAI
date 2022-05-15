from array import array
from struct import pack, unpack

import numpy as np


class PIC:
    def __init__(self, file_path=None):
        self.color_type = False                 #��ɫͼ����ǲ�ɫͼ��
        self.M = b"0"                           #�����ֶ�
        self.col = 512                          #ͼ������
        self.row = 512                          #ͼ������
        self.col_start = 0                      #ͼ�������
        self.row_start = 0                      #ͼ�������
        self.N = 0                              #�����ֶ�
        self.others = b""                       #����
        self.notes = b""                        #ע����Ϣ
        self.notes_size = len(self.notes)       #ע�����ֽ���

        self.image = np.array([])

    # ��bmp��Ϣ�����pic�ļ���
    def bmp2pic(self, bmp):
        self.col = bmp.width
        self.row = bmp.height

        self.image = bmp.image

        if bmp.image.shape[2] == 3:
            self.color_type = True
        else:
            self.color_type = False

    # ����pic�ļ�
    def save_pic(self,pic_path):
        f = open(pic_path, "wb")
        data = pack(f"?chhhhhh50s{self.notes_size}s", self.color_type, self.M, self.notes_size,
                    self.col, self.row, self.col_start, self.row_start, self.N, self.others, self.notes)
        f.write(data)
        rgb = self.image.ravel()
        for c in self.image:
            f.write(c)
        f.close()

    # ��ȡpic��ʽ��ͷ�ļ���Ϣ�Լ�rgb��Ϣ
    def load_pic(self, pic_path):
        f = open(pic_path, "rb")
        data = f.read(64)
        data = unpack(f"?chhhhhh50s{self.notes_size}s", data)
        self.pic_info(data)
        rgb_data = f.read()
        rgb_list = array('B', rgb_data)
        rgb_mat = np.reshape(rgb_list, (self.row, self.col, 3))

        return data, rgb_mat

    # ����ȡ����Ϣ����pic�ṹ��
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
