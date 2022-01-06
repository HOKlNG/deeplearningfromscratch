import numpy as np
import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
import time

"""
[해당 챕터 주요 개념]
- 활성함수 activation function
- 계단함수 step function
- 시그모이드함수 sigmoid function
- 렐루 Rectifield Linear Unit
- 순전파 forward Propagation
- 정규화 normalization
- 백색화 whitening

[numpy 주요 함수]
np.ndim(A)
np.shape
np.dot(A,B)
"""

#계단함수 step function
def step_function(x):
    y = x > 0
    return y.astype(np.int)

#시그모이드함수 sigmoid function
def sigmoid(x):
    return 1/ (1+np.exp(-x))

#렐루 Rectified Linear Unit
def relu(x):
    return np.maximum(0,x)

#sorft max apply overflow problem
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) +b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) +b3
    y = softmax(a3)
    return y

if __name__ == '__main__':
    #소프트맥스 총합 1 확인
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))

    """
    이미지확인
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28,28)

    print(img.shape)

    # img_show(img)

    """
    정확도확인인
    """
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0

    st = time.time()
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
        if p == t[i]:
            accuracy_cnt +=1
    print(time.time()-st)
    print("Accuracy: "+ str(float(accuracy_cnt)/ len(x)))

    """
    배치처리 (속도빠름)
    """
    batch_size = 100
    batch_accuracy = 0

    x, t = get_data()
    network = init_network()

    b_st = time.time()
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network,x_batch)
        p = np.argmax(y_batch, axis=1)
        batch_accuracy += np.sum(p == t[i:i+batch_size])

    print(time.time()-b_st)
    print("Accuracy batch: " + str(float(batch_accuracy)/ len(x)))

    # 속도차이 엄청나네...