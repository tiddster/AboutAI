import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
import cv2 as cv
from 噪声 import Noise

# 实际上是卷积, 利用卷积矩阵（高斯模板矩阵）在图像上进行 滑动窗口 的操作。
class GaussModel():
    def __init__(self, size):
        # 生成 size*size 的卷积矩阵
        self.gaussModel = np.zeros([size, size])
        self.pad = size // 2
        self.size = size

    def gaussFunction(self, x, y, sigma):
        # 高斯函数
        return math.exp( -(x**2 + y**2) / (2 * sigma**2) )  /  (2* sigma**2 * math.pi)

    #计算高斯模板
    def setGaussModel(self):
        #计算一个高斯模板
        pad = self.pad
        for i in range(0-pad, 0+pad):
            for j in range(0-pad, 0+pad):
                self.gaussModel[i+pad, j+pad] = self.gaussFunction(i, j, 0.8)
        # 归一化：计算总和
        sumOfMatrix = np.sum(self.gaussModel)
        # 归一化：计算权重
        for i in range(self.size):
            for j in range(self.size):
                self.gaussModel[i,j] /= sumOfMatrix

    #将图像再展开模板所对应的行列，用于处理边缘化数据
    def padding(self, row, column, layer):
        pad = self.pad
        paddedImg = np.zeros([row+2*pad, column+2*pad, layer])
        return paddedImg

    def gaussFilter(self, noiseImg):
        row, column, layer = noiseImg.shape
        #拓展图像
        paddedImg = self.padding(row,column,layer)
        #将noiseImg数据填充在tempImg中间
        paddedImg[self.pad:self.pad+row, self.pad:self.pad+column, :] = noiseImg[:,:,:]
        #用于记录最终结果
        resImg = noiseImg.copy()

        pad = self.pad

        #一二层循环：遍历像素点
        #三四层循环：计算利用高斯滤波处理后的数据
        for i in range(row):
            for j in range(column):
                tmp = 0
                for m in range(0 - pad, 0 + pad):
                    for n in range(0 - pad, 0 + pad):
                        tmp += paddedImg[i+m, j+n] * self.gaussModel[m+pad, n+pad]
                resImg[i, j,:] = tmp   #更新resImg数据
        return resImg
