import cv2
import numpy as np


def toGray(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # È¥³ýÔëÉù
    #blurredImg = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def plot(img):
    cv2.imshow('img',img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
