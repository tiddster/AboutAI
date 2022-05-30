import numpy as np
import pandas as pd

data = pd.read_csv('Student_Marks.csv')
n = len(data)
train_data = data[0: n // 10 * 7]
test_data = data[n // 10 * 7:]

# 将数据转换为矩阵形式
def convert(data):
    num_courses = data['number_courses']
    time_study = data['time_study']
    marks = data['Marks']
    return np.array(num_courses), np.array(time_study), np.matrix(marks).reshape(-1,1)

# 对数据进行增幂，相乘处理，否则拟合效果不好
def pre(data):
    x1, x2, y = convert(train_data)
    x1s, x2s, x1x2 = x1 ** 2, x2 ** 2, x1 * x2
    x1 = np.matrix(x1).reshape(-1, 1)
    x2 = np.matrix(x2).reshape(-1, 1)
    x1s = np.matrix(x1s).reshape(-1, 1)
    x2s = np.matrix(x2s).reshape(-1, 1)
    x1x2 = np.matrix(x1x2).reshape(-1, 1)
    ones = np.ones(x1.shape)
    X = np.hstack((ones, x1, x2, x1s, x2s, x1x2))
    return X, y

X, y = pre(train_data)
thetas = np.linalg.inv(X.T @ X) @ X.T @ y

# 假设函数
def h(X):
    return X @ thetas

#损失函数
def cost(X):
    m = len(X)
    return np.sum(np.square(h(X) - y)) / (2*m)

#利用测试集进行预测
X,y = pre(test_data)
print(f"当特征值为{thetas.ravel()}时, 损失值为{cost(X)}")

f = open('result.csv','w')
for x, y in zip(X,y):
    f.write(f"{np.array(h(x))[0][0]},{np.array(y)[0][0]} \n")

f.close()

