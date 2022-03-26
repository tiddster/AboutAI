import re

from TextRead import TextRead
from HMM_NER import HMM
from Viterbi_NER import Viterbi


def train(textFileName, ModelName):
    wordList, labelList = TextRead.readFile(textFileName)
    hmm = HMM(wordList, labelList)
    hmm.train(ModelName)
    return hmm


def getModel(textFileName, ModelName):
    wordList, labelList = TextRead.readFile(textFileName)
    hmm = HMM(wordList, labelList)
    hmm.getMatrix(ModelName)
    return hmm


if __name__ == '__main__':
    hmm = train("data\\math_train_data.txt", "NERModel.pkl")

    wordList, labelList = TextRead.readFile("data\\math_test_data.txt")
    text = []
    print(wordList)
    for list in wordList[26:29]:
        tmpText = ""
        for word in list:
            tmpText += word
        text.append(tmpText)

    v = Viterbi(hmm)
    for text in text:
        print(text)
        v.getPath(text)
        print(v.getRes(text))
