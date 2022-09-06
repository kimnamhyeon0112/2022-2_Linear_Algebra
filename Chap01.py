# 1번
def calc(n):
    sum=0
    for i in range(0,n):
        sum+=int(input())
    return sum

print("Input the number of values to be added => ")
count=int(input())
while count<=0:
    count=int(input())
print("Sum = " , calc(count))

# 2번
from gettext import npgettext
import numpy as np
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

v = np.array([[1],
              [2],
              [3]])

print("A = ",A)
print("v = ",v)

#3번
import numpy as np
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

v = np.array([[1],
              [2],
              [3]])

print("A = ",A)
print("v = ",v)

print()
print("A.shape = ", A.shape)
print("v.shape = ", v.shape)

w=np.array([[1,2,3]])
print()
print("w = ", w)
print("w.shape = ", w.shape)

B=np.array([[1,2,3],[4,5,6]])
print()
print("B = ", B)
print("B.shape = ", B.shape)