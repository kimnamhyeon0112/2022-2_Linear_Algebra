# 1번
# import numpy as np

# def getMinorMatrix(A,i,j):
#     n = len(A)
#     M = np.zeros((n-1, n-1))
#     for a in range(0, n-1):
#         k = a if (a < i) else a+1
#         for b in range(0, n-1):
#             l = b if (b < j) else b+1
#             M[a, b] = A[k, l]
#     return M

# def determinant(M):
#     if len(M) == 2:
#         return M[0,0]*M[1,1]-M[0,1]*M[1,0]
    
#     detVal = 0
#     for c in range(len(M)):
#         detVal += ((-1)**c)*M[0,c]*determinant(getMinorMatrix(M,0,c))
#     return detVal

# A = np.array([[-4,9,2,-1,0], [1,3,-3,-1,4], [2,0,1,3,0], [-2,1,-3,-1,5], [1,-5,1,0,5]])
# print("A= ", A)
# print("det(A) = ", determinant(A))
# print()

# 2번
# import numpy as np

# def confactor(A, i, j):
#     (n,m) = A.shape
#     M = np.zeros((n-1, m-1))
#     for a in range(0, n-1):
#         k = a if (a < i) else a+1
#         for b in range(0, m-1):
#             l = b if (b < j) else b+1
#             M[a,b] = A[k,l]
            
#     return (-1)**(i+j)*np.linalg.det(M)

# def inverseByAdjointMatrix(A):
#     detA = np.linalg.det(A)
#     (n,m) = A.shape
#     adjA = np.zeros((n, m))
    
#     for i in range(0,n):
#         for j in range(0, m):
#             adjA[j,i] = confactor(A, i, j)
#         if detA != 0.0:
#             return (1./detA) * adjA
#         else:
#             return 0
        
# A = np.array([[-4,9,2,-1,0], [1,3,-3,-1,4], [2,0,1,3,0], [-2,1,-3,-1,5], [1,-5,1,0,5]])
# print("A= ", A)
# Ainv = inverseByAdjointMatrix(A)
# print("A inverse = ", Ainv)

# 3번
import numpy as np

def solveByCramer(A, B):
    X = np.zeros(len(B))
    C = np.copy(A)
    for i in range(0, len(B)):
        for j in range(0, len(B)):
            C[j,i] = B[j]
            if i>0:
                C[j,i-1] = A[j,i-1]
        X[i] = np.linalg.det(C)/np.linalg.det(A)
    return X

A = np.array([[2,-1,5,1], [3,2,2,-6], [1,3,3,-1], [5,-2,-3,3]])
B = np.array([[-3], [-32], [-47], [49]])
X = solveByCramer(A, B)
print("A =", A)
print("B =", B)
print("X =", X)