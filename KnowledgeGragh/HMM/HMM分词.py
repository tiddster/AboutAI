#  对文本进行标记
import os.path
import re

def makeLabel(text):
    text_len = len(text)
    if text_len == 1:
        return "S"
    return "B" + "M" * (text_len - 2) + "E"


def textRead(textFile, labelFile):
    #在这里做出了修改，因为如果存在就要重新修改路径，很麻烦。改成如果存在则清空，数据
    if os.path.exists(labelFile):
        f = open(labelFile, "w", encoding="utf-8")
        f.truncate()
        f.close()

    # 分词的依据，在原来代码的基础上添加
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'

    #将file中的文件用\n进行切分放入list中
    allTextData = open(textFile, "r", encoding="utf-8").read().split("\n")
    print(allTextData)

    # with ... as ... 可以看作 是f = open(fileState, "w", encoding="utf-8") 并且 f.read()/f.write()
    with open(labelFile, "w", encoding="utf-8") as f:
        # tqdm是用于显示进度条的第三方库
        # enumerate是在遍历时可以同时给出下标和对应的值，类似于hashmap
        # 注意allTextData是列表
        for index, data in enumerate(allTextData):
            dataLabel = ""

            data = re.split(pattern, data)  # 在这里使用刚刚的pattern进行分词操作，不再仅仅依据空格分词
            for words in data:
                if words:
                    dataLabel += makeLabel(words) + " "

            if index != len(allTextData) - 1:
                dataLabel = dataLabel + "\n"  # 删去了split，用split会报错，split用于移除字符串首尾指定的字符，若未指定则移除空格或换行符

            f.write(dataLabel)

    print(open(labelFile, "r", encoding="utf-8").read().split("\n"))

if __name__ == "__main__":
    textRead('a.txt', 'g.txt')
