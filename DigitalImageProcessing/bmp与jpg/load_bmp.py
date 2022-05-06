import binascii
from struct import unpack

from DigitalImageProcessing import utils


def load_bmp_info(file):
    # 先将位图打开
    f = open(utils.lena_path, 'rb')  # 打开对应的文件
    '下面部分用来读取BMP位图的基础信息'
    f_type = str(f.read(2))  # 这个就可以用来读取 文件类型 需要读取2个字节
    file_size_byte = f.read(4)  # 这个可以用来读取文件的大小 需要读取4个字节
    f.seek(f.tell() + 4)  # 跳过中间无用的四个字节
    file_offset_byte = f.read(4)  # 读取位图数据的偏移量
    f.seek(f.tell() + 4)  # 跳过无用的两个字节
    file_wide_byte = f.read(4)  # 读取宽度字节
    file_height_byte = f.read(4)  # 读取高度字节
    f.seek(f.tell() + 2)  ## 跳过中间无用的两个字节
    file_bit_count_byte = f.read(4)  # 得到每个像素占位大小

    f_size, = unpack('l',file_size_byte)
    f_offset = unpack('l', file_offset_byte)
    f_wide, = unpack('l', file_wide_byte)
    f_height, = unpack('l', file_height_byte)
    f_bit_count = unpack('l', file_bit_count_byte)

    print(f_type)
    print(f_size, f_wide, f_height, f_bit_count, f_offset)


load_bmp_info(utils.biu_path)
