import numpy as np
from chapter03_neural_network.ch03_functions import softmax
from chapter04_nn_train.ch04_nn_train import cross_entropy_error

class MulLayer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

    def backward(self, dout):
        dx = dout*(1.0 - self.out) *self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/ batch_size
        return dx



if __name__ == "__main__":

    """
    MulLayer
    """
    print('MulLayer')
    apple = 100
    apple_num = 2
    tax = 1.1

    # ?????????
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # ?????????
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, tax)

    print(' ')

    print('??????, ?????????')
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # ?????????

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    #?????????
    apple_price = mul_apple_layer.forward(apple,apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)

    #?????????
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)
    print('---------------------------------------')

    print('ReLu')
    x = np.array( [[1.0, -0.5], [-2.0, 3.0]])
    print(x)

    mask = (x<=0)
    print(mask)
    print('---------------------------------------')
