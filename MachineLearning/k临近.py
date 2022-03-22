from numpy import *
import numpy as np
import matplotlib.pyplot as plt

"""
算法 根据距离来 判断某个未分类数据 属于哪一个集合
"""


# 导入数据
def createDataSet():
    gourp = array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return gourp, labels


"""
KNN伪代码：
1. 计算 目标点 与 已知点 的距离
2. 将计算出来的距离排序
3. 统计出距离最小的k个点
4. 判断这k个点所属的标签，观察最高频率的标签
5. 给未知数据贴上标签
"""


def classify(inX, dataSet, labels, k):
    # .shape 函数是获取长度; 这里是获取dataSet第0行的长度
    dataSetSize = dataSet.shape[0]

    # TODO：第一步，计算距离
    # 例: tile(a, (1,2)) 将a数组在 行/x轴方向 复制1倍， 在列/y轴方向 复制2倍
    diffMatrix = tile(inX, (dataSetSize, 1)) - dataSet  # ->现在diffMat数组中对应的就是被减去后的差值
    squareMatrix = diffMatrix ** 2
    sqDis = squareMatrix.sum(axis=1)
    distances = sqDis ** 0.5

    print(distances)

    # TODO: 第二步， 排序
    # argsort() 是将数组排序后，返回索引值
    sortedDistances = distances.argsort()

    # TODO： 第三,四步，选择距离最小的k个点,并统计标签频率
    labelCount = {}
    for i in range(k):
        label = labels[sortedDistances[i]]
        # 字典的get函数, 获取key为label的value值;当label不存在时，则将label作为key, 0作为value存入字典
        labelCount[label] = labelCount.get(label, 0) + 1

    # TODO: 第五步，选出频率最高的并排序
    sortedPointCount = sorted(labelCount)

    return sortedPointCount[0][0]

if __name__ == '__main__':
    dataSet1 = np.random.rand(100,2)
    dataSet2 = np.random.rand(100,2)
    labels1 = ['A'] * 100
    labels2 = ['B'] * 100
    aimSet = np.random.rand(2)

    print(aimSet)

    # 矩阵竖向连接vertical，同理横向是horizontal
    totalDataSet = np.vstack((dataSet1,dataSet2))
    labels = np.hstack((labels1,labels2))

    aimLabel = classify(aimSet, totalDataSet, labels, 2)

    x1, y1 = [],[]
    x2, y2 = [],[]
    for i in range(len(dataSet1)):
        x1.append(dataSet1[i][0])
        y1.append(dataSet1[i][1])

    for i in range(len(dataSet2)):
        x2.append(dataSet2[i][0])
        y2.append(dataSet2[i][1])

    plt.scatter(x1, y1, c='b')
    plt.scatter(x2, y2, c='g')
    for i in range(0, len(dataSet1)):
        plt.text(x1[i],y1[i], labels1[i])
    for i in range(0, len(dataSet2)):
        plt.text(x2[i], y2[i], labels2[i])

    plt.scatter(aimSet[0], aimSet[1], c='r')
    plt.text(aimSet[0], aimSet[1], aimLabel)
    plt.show()

