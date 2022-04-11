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

def initTestData(textFileName):
    wordList, labelList = TextRead.readFile(textFileName)
    textList = []
    for i in range(730):
        tmpText = ""
        if wordList[i] is not None:
            for word in wordList[i]:
                tmpText += word
            readLabels.append(labelList[i])
            textList.append(tmpText)
    return textList

def TestContext(textFileName):
    textList = open(textFileName, "r", encoding="utf-8").read().split("\n")
    return textList


if __name__ == '__main__':
    hmm = getModel("data\\math_train_data.txt", "NERModel.pkl")

    readLabels, predictLabels = [], []

    textList = initTestData("data\\math_test_data.txt")

    allEntities = {}

    # 获取预测的标签列表
    v = Viterbi(hmm)
    for text in textList:
        (tempText, tempLabels,entities) = v.getPath(text)
        for entity in entities:
            allEntities[entity] = allEntities.get(entity, 0) + 1
        predictLabels.append(tempLabels)

    # 评估
    assess = Assess(readLabels, predictLabels, hmm)
    print(assess.CM)
    for label in hmm.index_to_label:
        print(f"{label}的F1-SCORE为{assess.F1_score(label)}")

    f = open("data\\result.txt","w")
    f.truncate()
    for key, value in allEntities.items():
        f.write(f"{key}:{value} \n")
    f.close()

