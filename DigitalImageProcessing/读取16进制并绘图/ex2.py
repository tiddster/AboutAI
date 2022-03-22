import numpy as np
from numpy import shape
from skimage import io
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt

img = imread("ex2.bmp")
gray = rgb2gray(img)
gray *= 255
gray = gray.astype(np.uint8)
print(gray)

m,n = img.shape[0], img.shape[1]
gray = gray.reshape(1,m*n)
line = ['%X' % i for i in gray[0]]

file = open("ex2.txt",'w')
for i in range(m*n):
    file.write(line[i]+'\n')
file.close()

loadFile = open("ex2.txt",'r')
loadStr = loadFile.read().splitlines()

res = np.zeros([m,n])
for i in range(m):
    for j in range(n):
        res[i,j] = int(loadStr[i*m+j],16)
print(res)

res = res.astype(np.uint8)
io.imshow(res)
plt.show()