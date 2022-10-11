# 1번
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
    
# A = np.array([[1., 2.], [3., 4.]])
# pprint("A", A)

# Ainv1 = np.linalg.matrix_power(A, -1)
# pprint("linalg.matrix_power(A, -1) => Ainv1", Ainv1)

# Ainv2 = np.linalg.inv(A)
# pprint("np.linalg.inv(A) => Ainv2", Ainv2)

# pprint("A*Ainv1", np.matmul(A, Ainv1))
# pprint("A*Ainv2", np.matmul(A, Ainv2))

# B = np.random.rand(3,3)
# pprint("B =", B)
# Binv = np.linalg.inv(B)
# pprint("Binv =", Binv)
# pprint("B*Binv =", np.matmul(B, Binv))

# C = np.array([[5,3,2,1],[6,2,4,5],[7,4,1,3],[4,3,5,2]])
# D = np.array([[4],[2],[5],[1]])
# x = np.matmul(np.linalg.inv(C), D)
# pprint("x", x)
# pprint("C*x", np.matmul(C, x))

# 2번
import numpy as np

def LU(A):
    (n,m) = A.shape
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    
    for i in range(0, n):
        for j in range(i, n):
            U[i, j] = A[i, j]
            for k in range(0, i):
                U[i, j] = U[i , j] - L[i, k] * U[k, j]
        L[i,i] = 1
        if i < n-1:
            p = i + 1
            for j in range(0, p):
                L[p, j] = A[p, j]
                for k in range(0, j):
                      L[p, j] = L[p, j] - L[p, k] * U[k, j]
                L[p, j] = L[p, j] / U[j, j]
    return L, U
def LUSolver(A, b):
    L, U = LU(A)
    n = len(L)
    y=np.zeros((n,1))
    for i in range(0,n):
        y[i]=b[i]
        for k in range(0,i):
            y[i]-=y[k]*L[i,k]
    x=np.zeros((n,1))
    for i in range(n-1, -1, -1):
        x[i]=y[i]
        if i<n-1:
            for k in range(i+1, n):
                x[i]-=x[k]*U[i,k]
        x[i]=x[i]/float(U[i,i])
    return x

A = np.array([[5,3,2,1],[6,2,4,5],[7,4,1,3],[4,3,5,2]])
b = np.array([[4],[2],[5],[1]])

L,U = LU(A)
pprint("A", A)
pprint("L", L)
pprint("U", U)

x = LUSolver(A,b)
pprint("x", x)
        