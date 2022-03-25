from HMM import HMM
from HMM分词 import *
from Viterbi import Viterbi

if __name__ == '__main__':
    print("开始读取数据......")
    textRead('a.txt', 'g.txt')
    print("数据处理完毕!")
    hmm = HMM('a.txt', 'g.txt')
    print("开始训练HMM模型......")
    hmm = HMM('a.txt', 'g.txt')
    hmm.train("model.txt")
    print("模型训练结束!\n")

    print("开始读取数据......")
    textRead("C:\\Users\\tiddler\Desktop\\0322课程\\0322课程\\word_cut-dataset\\test_text.txt", 'g.txt')
    print("数据处理完毕!")
    hmm = HMM("C:\\Users\\tiddler\Desktop\\0322课程\\0322课程\\word_cut-dataset\\test_text.txt", 'g.txt')
    print("开始训练HMM模型......")
    hmm = HMM("C:\\Users\\tiddler\Desktop\\0322课程\\0322课程\\word_cut-dataset\\test_text.txt", 'g.txt')
    hmm.train("model.txt")
    print("模型训练结束!\n")

    hmm.getMatrix("model.txt")

    text = "“抓好党建是最大的政绩”。近年来，我市始终坚持以党的建设统揽全局工作，以习近平新时代中国特色社会主义思想为指导，全面贯彻新时代党的建设总要求和新时代党的组织路线，不断提升党的建设质量，扎实推进全面从严治党向纵深发展，进一步深化基层作风整治，着力营造干事创"
    viterbi = Viterbi(hmm)
    print(viterbi.dealText(text))