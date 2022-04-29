import cv2


# ��ֵ�˲�
def averageBlur(img, a=3):
    blur = cv2.blur(img, (a, a))
    return blur


# �����˲�
def boxBlur(img, a=3):
    blur = cv2.boxFilter(img, -1, (a, a), normalize=True)
    return blur


# ��˹�˲�
def gaussBlur(img, a=5):
    blur = cv2.GaussianBlur(img, (a, a), 1)
    return blur


# ��ֵ�˲�
def medianBlur(img, a=5):
    blur = cv2.medianBlur(img, a)
