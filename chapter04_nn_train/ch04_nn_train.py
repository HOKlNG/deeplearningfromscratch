import numpy as np

"""
4.2 손실함수
신경망 서응의 나쁨을 나타내는 지표로 작아야 좋음
"""

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
#수치미분
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

# 편미분
def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

#기울기
def numeriacl_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)계산
        x[idx] = tmp_val + h
        fxh1=f(x)

        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(X)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

if __name__ == '__main__':
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

    print('2 일 확률이 가장 높다고 추정(0.6)')
    print(sum_squares_error(np.array(y), np.array(t)))


    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print('7 일 확률이 가장 높다고 추정(0.6)')
    print(sum_squares_error(np.array(y), np.array(t)))

    print('-----------------------------------------')
    print('cross entropy_error')
    print('2 일 확률이 가장 높다고 추정(0.6)')
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print('7 일 확률이 가장 높다고 추정(0.6)')
    print(cross_entropy_error(np.array(y), np.array(t)))

    print('-----------------------------------------')
    print('수치미분')
    print(numerical_diff(function_1,5))
    print(numerical_diff(function_1,10))

    print('-----------------------------------------')
    print('편미분')
    print(numerical_diff(function_tmp1,3.0))
    print(numerical_diff(function_tmp2,4.0))