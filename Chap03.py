# 1번
import numpy as np

print("벡터의 결합에 의한 행렬 생성")
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
v3 = np.array([7,8,9])

A = np.stack([v1,v2,v3])
print("A= ", A)

B = np.column_stack([v1,v2,v3])
print("B= ", B)

C = np.array([[1,2],[3,4],[5,6]])
print("C= ", C)

D = np.column_stack([C, v3])
print("D= ", D)

print("행렬의 성분 접근")
E= np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print("E[0,3] =", E[0,3])
print("E[1,2] =", E[1,2])

print("E[0:2, 2] =", E[0:2, 2])
print("E[0:2, 0:4] =", E[0:2, 0:4])
print("E[2, :] =", E[2, :])

print("성분의 변경")
print("E =", E)

print("E[0,0] = ", E[0,0])
E[0,0] = -1
print(E)
print("E[0,0] = ", E[0,0])

# 2번
import numpy as np

def pprint(msg, A):
    print("---", msg, "---")
    (n,m) = A.shape
    for i in range(0, n):
        line = ""
        for j in range(0, m):
            line += "{0:.2f}".format(A[i,j]) + "\t"
        print(line)
    print("")

A = np.array([[1., 2.], [3., 4.]])
B = np.array([[2., 2.], [1., 3.]])
C = np.array([[4., 5., 6.], [7., 8., 9.]])
V = np.array([[10.], [20.]])

pprint("A+B", A+B)
pprint("A-B", A-B)

pprint("3*A", 3*A)
pprint("2*V", 2*V)

pprint("matmul(A,B)", np.matmul(A,B))
pprint("matmul(A,C)", np.matmul(A,C))
pprint("A*V", A*V)

pprint("matrix_power(A, 2)", np.linalg.matrix_power(A, 2))
pprint("matrix_power(A, 3)", np.linalg.matrix_power(A, 3))

pprint("A*B", A*B)
pprint("A/B", A/B)
pprint("A**2 == A*A", A**2)

pprint("A.T", A.T)
pprint("V.T", V.T)

M = np.diag([1,2,3])
pprint("diag(1,2,3) =", M)

D11 = np.array([[1, 2], [3, 4]])
D12 = np.array([[5], [6]])
D21 = np.array([[7, 7]])
D22 = np.array([[8]])
D = np.block([[D11, D12], [D21, D22]])
pprint("block matrix", D)