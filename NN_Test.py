import numpy as np
from NeuralNetwork import NeuralNetwork


N = 10
P = 3
layers = [4, 1]

X = np.random.uniform(0, 1, size=(N,P))
y = np.random.uniform(0, 1, size=(N,1))

nn = NeuralNetwork(X, y, layers)





