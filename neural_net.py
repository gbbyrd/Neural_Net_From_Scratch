import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Create a 3 layer neural network and test it on the mnist dataset


""" 
Architecture

Input: 784 Neurons

Hidden: 20 Neurons

Ouput: 10 Neurons

Activation Function: Sigmoid

Loss Function: MSE

"""

# load the mnist dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.resize(train_X, (60000, 784))
train_X = train_X.T
train_X = train_X / 255
test_X = np.resize(test_X, (10000, 784))
test_X = test_X.T
test_X = test_X / 255

def init_parameters():
    w1 = np.random.rand(30, 784) - 0.5
    b1 = np.random.rand(30, 1) - 0.5
    w2 = np.random.rand(10, 30) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return w1, b1, w2, b2

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    
    return z1, a1, z2, a2

def d_ReLU(x):
    yes = x > 0
    yes = yes.astype(int)
    return yes

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1 - sig)

def one_hot(Y):
    one_hot = np.zeros((10, Y.size))
    for idx in range(Y.size):
        index = Y[idx]
        one_hot[index ,idx] = 1
    return one_hot

def MSE(pred, y):
    one_hot_y = one_hot(y)
    mse = np.sum(0.5*(one_hot_y-pred)**2)
    return mse

def d_MSE(pred, y):
    one_hot_y = one_hot(y)
    return -(one_hot_y - pred)

def backprop(x, y, w1, b1, w2, b2, z1, a1, z2, a2):
    m = len(x[0])
    dz2 = d_MSE(a2, y)
    dw2 = (1/m) * np.dot(dz2, a1.T)
    sum_dz2 = np.sum(dz2, axis=1)
    sum_dz2 = np.resize(sum_dz2, (len(sum_dz2), 1))
    db2 = (1/m) * sum_dz2
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * d_ReLU(z1)
    dw1 = (1/m) * np.dot(dz1, x.T)
    sum_dz1 = np.sum(dz1, axis=1)
    sum_dz1 = np.resize(sum_dz1, (len(sum_dz1), 1))
    db1 = (1/m) * sum_dz1
    
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr=.1):
    w1 = w1 - lr*dw1
    b1 = b1 - lr*db1
    w2 = w2 - lr*dw2
    b2 = b2 - lr*db2
    
    return w1, b1, w2, b2

def get_pred(pred):
    _pred = np.argmax(pred, axis=0)
        
    return _pred

def print_sizes(arrays):
    for arr in arrays:
        print(arr.shape)
    
    

def gradient_descent(epochs, train_X, train_Y, test_X, test_Y):
    
    w1, b1, w2, b2 = init_parameters()
    
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(train_X, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backprop(train_X, train_Y, w1, b1, w2, b2, z1, a1, z2, a2)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2)
        
        if epoch % 10 == 0:
            loss, accuracy = test(test_X, test_Y, w1, b1, w2, b2)
            print(f'Epoch: {epoch+1}\nLoss: {loss}\nAccuracy: {accuracy}\n----------------------------------------')
            
def test(x, y, w1, b1, w2, b2):
    z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
    loss = MSE(a2, y)
    pred = get_pred(a2)
    total_correct = sum(pred == y)
    total = len(x[0])
    return loss/total, total_correct/total
    
if __name__ == '__main__':
    gradient_descent(500, train_X, train_y, test_X, test_y)
    
        
































# # Define the functions
# def init_params():
#     w1 = np.random.rand(20, 784) - 0.5
#     b1 = np.random.rand(20, 1) - 0.5
#     w2 = np.random.rand(10, 20) - 0.5
#     b2 = np.random.rand(10, 1) - 0.5
    
#     return w1, b1, w2, b2

# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#     # return np.maximum(x, 0)

# def softmax(x):
#     return np.exp(x)/sum(np.exp(x))

# def forward(w1, b1, w2, b2, x):
#     z1 = np.dot(w1, x) + b1
#     a1 = sigmoid(z1)
#     z2 = np.dot(w2, a1) + b2
#     a2 = softmax(z2)
    
#     return z1, a1, z2, a2

# def d_sigmoid(x):
#     sig = sigmoid(x)
#     return sig*(1 - sig)
#     # return x > 0

# def one_hot(Y):
#     one_hot = np.zeros((10, Y.size))
#     for idx in range(Y.size):
#         index = Y[idx]
#         one_hot[index ,idx] = 1
#     return one_hot

# def MSE(x, y):
#     one_hot_y = one_hot(y)
#     mse = np.sum((one_hot_y - x)**2)
#     print(mse)
#     return mse

# def d_MSE(x, y):
#     one_hot_y = one_hot(y)
#     return -(one_hot_y-x)

# def backprop(z1, a1, z2, a2, w2, x, y):
#     dz2 = d_MSE(a2, y)
#     dw2 = (1/y.size) * np.dot(dz2, a1.T)
#     db2 = (1/y.size) * np.resize(np.sum(dz2, axis=1), (10, 1))
#     da1 = np.dot(w2.T, dz2)
#     dz1 = da1 * d_sigmoid(z1)
#     dw1 = (1/y.size) * np.dot(dz1, x.T)
#     db1 = (1/y.size) * np.resize(np.sum(dz1, axis=1), (20, 1))
    
#     return dw1, db1, dw2, db2

# def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
#     w1 = w1 - lr*dw1
#     b1 = b1 - lr*db1
#     w2 = w2 - lr*dw2
#     b2 = b2 - lr*db2
    
#     return w1, b1, w2, b2

# def argmax_output(output):
#     argmax = np.argmax(output, axis=0)
#     return argmax

# def gradient_descent(train_x, train_y, test_x, test_y, lr, epochs):
#     w1, b1, w2, b2 = init_params()
    
#     for epoch in range(epochs):
#         z1, a1, z2, a2 = forward(w1, b1, w2, b2, train_X)
#         dw1, db1, dw2, db2 = backprop(z1, a1, z2, a2, w2, train_X, train_y)
#         w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)
        
#         if epoch % 5 == 0:
#             loss, accuracy = test(w1, b1, w2, b2, test_x, test_y)
#             print(f'-----------------------------------------------------------')
#             print(f'Epoch {epoch+1}')
#             print(f'Loss {loss} ------- Accuracy {accuracy}')

# def test(w1, b1, w2, b2, test_x, test_y):
#     z1, a1, z2, a2 = forward(w1, b1, w2, b2, test_x)
#     loss = MSE(a2, test_y)
#     argmax_a2 = argmax_output(a2)
#     correct = np.sum(argmax_a2 == test_y)
    
#     return loss/test_y.size, correct/test_y.size
    
# if __name__=='__main__':
    
#     lr = 0.2
#     epochs = 500
    
#     w1, b1, w2, b2 = gradient_descent(train_X, train_y, test_X, test_y, lr, epochs)

    
    