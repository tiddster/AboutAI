import numpy as np

# 利用numpy.linalg.norm()计算向量v的三种范数
v = np.array([[9,7,8,6]])
print(v)

print(f"向量的1-范数 {np.linalg.norm(v,1)}")
print(f"向量的2-范数 {np.linalg.norm(v,2)}")
print(f"向量的无穷范数 {np.linalg.norm(v, np.inf)}")


# 利用numpy.linalg.norm()计算矩阵的四种范数
m = np.random.rand(3,3)
print(m)

print(f"矩阵的1-范数 {np.linalg.norm(m,1)}")
print(f"矩阵的2-范数 {np.linalg.norm(m,2)}")
print(f"矩阵的无穷范数 {np.linalg.norm(m, np.inf)}")
print(f"矩阵的F-范数 {np.linalg.norm(m, 'fro')}")