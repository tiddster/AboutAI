import re

from TextRead import TextRead
from HMM_NER import HMM
from Viterbi_NER import Viterbi
from Assess import Assess


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

    readLabels, predictLabels = [], []

    # 初始化单词列表和标签列表
    wordList, labelList = TextRead.readFile("data\\math_test_data.txt")
    text = []
    for i in range(780):
        tmpText = ""
        if wordList[i] is not None:
            for word in wordList[i]:
                tmpText += word
            readLabels.append(labelList[i])
            text.append(tmpText)

    # 获取预测的标签列表
    v = Viterbi(hmm)
    for text in text:
        (tempText, tempLabels) = v.getPath(text)
        predictLabels.append(tempLabels)
    print(readLabels,predictLabels)

    # 评估
    assess = Assess(readLabels, predictLabels, hmm)
    print(assess.CM)
    for label in hmm.index_to_label:
        print(f"{label}的F1-SCORE为{assess.F1_score(label)}")

