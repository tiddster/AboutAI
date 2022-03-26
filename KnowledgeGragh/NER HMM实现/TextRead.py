class TextRead():
    def __init__(self, textFile):
        self.file = textFile

    @staticmethod
    def readFile(textName):
        # 最终读取的列表，将整个文章中文字和标签分开的列表
        wordList = []
        labelList = []

        # 句子列表，将整个句子中的文字和标签分开的列表
        sentenceWord = []
        sentenceLabel = []

        # 关键词集合
        keyWordList = []
        tmpKeyWord = ""

        all_data = open(textName, "r", encoding="utf-8").read().split("\n")
        for index, text in enumerate(all_data):
            #print(index, text)
            if text:
                # 因为在数据集中，文字和标签两两对应，拆开后必然是2，若不是2则不符合要求
                if len(text.strip('\n').strip('\r').split(" ")) != 2:
                    continue
                x, y = text.strip('\n').strip('\r').split(" ")
                #print(x+"and"+y)

                if y == "B-MAT" or y == "I-MAT":
                    tmpKeyWord += x
                elif y == "O" and tmpKeyWord:
                    keyWordList.append(tmpKeyWord)
                    tmpKeyWord = ""

                sentenceWord.append(x)
                sentenceLabel.append(y)
            else:
                wordList.append(sentenceWord)
                labelList.append(sentenceLabel)
                sentenceWord = []
                sentenceLabel = []
        wordList.append(sentenceWord)
        labelList.append(sentenceLabel)
        #print(keyWordList)
        return wordList, labelList

if __name__ == '__main__':
    textName = 'math_train_data.txt'
    TextRead.readFile(textName)