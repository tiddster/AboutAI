import numpy as np

x = np.array([[2,1,-3,4]])
xT = x.T

print(f"向量X的1-范数{np.linalg.norm(xT, 1)}")
print(f"向量X的2-范数{np.linalg.norm(xT, 2)}")
print(f"向量X的无穷范数{np.linalg.norm(xT, np.inf)}")

A = np.array([
    [1,1,1,1],
    [-1,1,-1,1],
    [-1,-1,1,1],
    [1,-1,-1,1]
])

print(f"向量A的1-范数{np.linalg.norm(A, 1)}")
print(f"向量A的2-范数{np.linalg.norm(A, 2)}")
print(f"向量A的无穷范数{np.linalg.norm(A, np.inf)}")
print(f"向量A的F-范数{np.linalg.norm(A, 'fro')}")