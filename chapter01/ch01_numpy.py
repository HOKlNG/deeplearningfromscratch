"""
chapter 1.5 넘파이
넘파이 설치할 것.
"""

import numpy as np
print(' ')
print('1.5.2')
x = np.array([1.0, 2.0, 3.0])
print(x)


####################################
print(' ')
print('1.5.3')

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x/2.0)

####################################
print(' ')
print(' -- 1.5.4 Numpy N차원 배열')
print(' ')

A = np.array([[1,2],[3,4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3,0], [0,6]])

print(A+B)
print(A*B)
print(A*10)


####################################
print(' ')
print(' -- 1.5.5 브로드캐스트')
print(' ')

A = np.array([[1,2], [3,4]])
B = np.array([[10,20]])
print(A*B)

####################################
print(' ')
print(' -- 1.5.6 원소 접근')
print(' ')

X = np.array([[51,55], [14,19], [0,4]])
print(X)

print(X[0])

for row in X:
    print(row)

X = X.flatten() #1차원으로 변화ㅏㄴ
print(X)

print(X[np.array([0,2,4])]) #인덱스가 0, 2, 4 인 원소 얻기

print(X>15)

print(X[X>15])