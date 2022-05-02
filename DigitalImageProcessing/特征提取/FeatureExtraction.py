import cv2


def featureExtraction(img):
    img = cv2.Canny(img,200,300)
    return img
