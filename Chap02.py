# 1번
import numpy as np

a = np.zeros((2, 3))    # 2x3 영행렬
print("a = \n", a)

b = np.ones((2, 2))     # 2x2 행렬 모든성분 1
print("b = \n", b)

c = np.full((3, 2), 3)  # 3x2 행렬 모든성분 3
print("c = \n", c)

d = np.eye(2)           # 2x2 단위행렬
print("d = \n", d)

# 2번
import numpy as np
#행렬 A 출력
def pprint(msg, A):
    print("---", msg, "---")
    (n,m) = A.shape
    for i in range(0, n):
        line = ""
        for j in range(0, m):
            line += "{0:.2f}".format(A[i,j]) + "\t"
            if j == n-1:
                line += "| "
        print(line)
    print("")
    
#Gauss-Jordan 소거법 수행함수
def gauss(A):
    (n,m) = A.shape
    
    for i in range(0, min(n,m)):
        # i번째 열에서 절댓값이 최대인 성분의 행 선택
        maxEl = abs(A[i,i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k,i]) > maxEl:
                maxEl = abs(A[k,i])
                maxRow = k
                
        # 현재 i번째 행과 최댓값을 갖는 행 maxRow의 교환
        for k in range(i, m):
            # tmp = A[maxRow,k]
            # A[maxRow,k] = A[i,k]
            # A[i,k] = tmp
            
            A[maxRow, k], A[i,k] = A[i,k], A[maxRow, k]
           
        # 추축성분 1로 만들기
        piv = A[i,i]
        for k in range(i, m):
            A[i,k] = A[i,k]/piv
            
        # 현재 i번째 열의 i번째 행을 제외한 모두 성분을 0으로 만들기
        for k in range(0, n):
            if k != i:
                c = A[k,i]/A[i,i]
                for j in range(i, m):
                    if i == j:
                        A[k,j] = 0
                    else:
                        A[k,j] = A[k,j] - c * A[i,j]
        pprint(str(i+1)+"번째 반복", A) # 중간과정 출력
            
    # Ax=b의 해 반환
    x = np.zeros(m-1)
    for i in range(0, m-1):
        x[i] = A[i,m-1]
    return x
    
# 주어진 문제 풀이
A = np.array([[2., 2., 4., 18.],[1., 3., 2., 13.], [3., 1., 3., 14.]])

pprint("주어진 문제", A)
x = gauss(A)

# 출력
(n,m) = A.shape
line = "해:\t"
for i in range(0, m-1):
    line += "{0:.2f}".format(x[i]) + "\t"
print(line)
            