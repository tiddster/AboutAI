import cv2


# 均值滤波
def averageBlur(img, a=3):
    blur = cv2.blur(img, (a, a))
    return blur


# 方框滤波
def boxBlur(img, a=3):
    blur = cv2.boxFilter(img, -1, (a, a), normalize=True)
    return blur


# 高斯滤波
def gaussBlur(img, a=5):
    blur = cv2.GaussianBlur(img, (a, a), 1)
    return blur


# 中值滤波
def medianBlur(img, a=5):
    blur = cv2.medianBlur(img, a)
