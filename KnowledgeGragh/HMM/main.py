from HMM import HMM
from HMM分词 import *
from Viterbi import Viterbi

if __name__ == '__main__':
    print("开始读取数据......")
    textRead('a.txt', 'g.txt')
    print("数据处理完毕!")
    hmm = HMM('a.txt', 'g.txt')
    print("开始训练HMM模型......")
    hmm.train()
    print("模型训练结束!\n")

    text = "时光的列车慢慢前进，不知道在哪里停靠，白天与黑夜不慌不忙的交替，不知道蹉跎了谁的岁月"
    viterbi = Viterbi(hmm)
    print(viterbi.getPath(text))