"""
perceptron

2.1 퍼셉트론이란.?
다수의 신호를 입력받아 하나의 신호를 출력함
신호
흐름
뉴런
노드
가중치
임계값

2.2 단순한 논리회로
AND게이트
NAND게이트
OR게이트

2.3 퍼셉트론 구현하기
"""

#2.3.1 간단한구현

def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2*w2

    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
print('2.3.1')
print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))
print('---------------')

#2.3.2 가중치 편향 도입

import numpy as np
x = np.array([0,1]) #입력
w = np.array([0.5, 0.5]) #가중치
b = -0.7 #편향
print(w*x)
print('-----------')
print(np.sum(x*w))
print(np.sum(w*x)+b)

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7

    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else :
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7

    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else :
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2

    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else :
        return 1

"""
2.4 퍼셉트론의 한계
배타적 논리 합
XOR게이트
선형
비선형
"""

"""
2.5 다층 퍼셉트론이 충돌한다면
"""
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1, s2)
    return y

print('2.5.2 XOR GATE')
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
print('-------------')
"""
단층으로 표현하지 못한 것들을 층을 늘려 구현할 수 있게 됨
"""

"정리"
"""
- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다.
- 퍼셉트론은 '가중치'와 '편향'을 매개변수로 설정한다.
- 퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다.
- XOR 게이트는 단층 퍼셉트론으로 표현할 수 없다.
- 2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있다.
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.
- 다층 퍼셉트론은(이론상) 컴퓨터를 표현할 수 있다.
"""