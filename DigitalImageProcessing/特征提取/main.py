import DataProcess as TG
import FeatureExtraction as FE
import torch

if __name__ == '__main__':
    img = TG.toGray('biu.jpg')
    img = FE.featureExtraction(img)
    TG.plot(img)

