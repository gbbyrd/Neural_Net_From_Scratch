import matplotlib.pyplot as plt
import numpy as np

dataset = [[2.7810836,2.550537003,0],
 [1.465489372,2.362125076,0],
 [3.396561688,4.400293529,0],
 [1.38807019,1.850220317,0],
 [3.06407232,3.005305973,0],
 [7.627531214,2.759262235,1],
 [5.332441248,2.088626775,1],
 [6.922596716,1.77106367,1],
 [8.675418651,-0.242068655,1],
 [7.673756466,3.508563011,1]]

dataset = np.array(dataset)
# Plot the data to determine if linearly separable
fig, axis = plt.subplots(1, 1)
axis.scatter(dataset[:5,0], dataset[:5,1], c='red')
axis.scatter(dataset[5:,0], dataset[5:,1], c='blue')


w = np.random.rand(1)
b = np.random.rand(1)

def get_prediction(x, w, b):
    activation = w*x + b
    pred = activation > 0
    return pred

def get_delta_w(pred, target, x):
    delta_w = (target-pred)*x
    return delta_w

def get_delta_b(pred, target):
    delta_b = (target-pred)
    return delta_b

def make_updates(w, b, delta_w, delta_b, lr):
    w = w + lr*delta_w
    b = b + lr*delta_b
    
    return w, b

w = -w

pred = []
for data in dataset:
    pred1 = get_prediction(data[0], w, b)
    pred.append(pred1)
    
print(f'Initial prediction: {pred}')
        

for epoch in range(25):
    for data in dataset:
        pred = get_prediction(data[0], w, b)
        delta_w = get_delta_w(pred, data[2], data[0])
        delta_b = get_delta_b(pred, data[2])
        w, b = make_updates(w, b, delta_w, delta_b, .1)
        
pred = []
for data in dataset:
    pred1 = get_prediction(data[0], w, b)
    pred.append(pred1)
    
print(f'Final predication: {pred}')
        

