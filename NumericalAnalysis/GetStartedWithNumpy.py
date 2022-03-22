import numpy as np

# 1.构建一维
arr1 = np.array([1,2,3,4])
print(f"一维数组: {arr1}")

# 2.构建二维
arr2 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(f"二维数组: {arr2}")

# 3.计算点积
arr3 = np.array([
    [9,8,7],
    [6,5,4],
    [3,2,1]
])
arr4 = np.dot(arr2,arr3)
print(f"乘积为 {arr4}")

#合并矩阵
temp = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])
arrMergeColumn = np.hstack((arr2,temp))
arrMergeRow = np.vstack((arr3, temp))

print(f"合并矩阵-列：{arrMergeColumn}" + "\n"
      f"合并矩阵-行：{arrMergeRow}")

# 矩阵转置
arr5 = arr2.T
print(f"矩阵转置为 {arr5}")

# 创造一个pi常量数组
arr6 = np.full((2,2),3.14)
print(f"pi常量矩阵 {arr6}")

# 创造一个3*3单位矩阵
arr7 = np.eye(3)
print(f"单位矩阵\n {arr7}")

# 创造一个随机数组
arr8 = np.random.randn(2,2)
print(f"random array {arr8}")

#
A = np.arange(0,30,5) # 创建[0,5,10,15,20,25],第三个参数为步长
B = np.linspace(0,30,6) # 创建[0,6,12,18,24,30], 第三个参数为元素个数
print(A,B)