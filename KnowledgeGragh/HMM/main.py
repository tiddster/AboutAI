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

    text = "时光的列车慢慢前进,不知道在哪里停靠,白天与黑夜不慌不忙的交替,不知道蹉跎了谁的岁月。全心全意跟党走,做新时代全面发展新青年"
    viterbi = Viterbi(hmm)
    print(viterbi.dealText(text))