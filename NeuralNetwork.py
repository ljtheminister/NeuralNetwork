import numpy as np
import random
from numpy import sqrt
from activation_functions import *
from cost_functions import *



class NeuralNetwork:
    def __init__(self, X, y, layers, alpha=0.1, test_prop=0.9, seed_parameter=1, activation_function='logistic', loss_function='squared_loss', output_function='linear'):

	self.X = X
	self.y = y
	self.N, self.P = X.shape

	self.seed = seed_parameter
	self.layers = layers
	
	self.N_train = np.floor(self.N*(1-test_prop))
	self.N_test = self.N - self.N_train

	row_idx = [i for i in xrange(self.N)]
	random.shuffle(row_idx)
	'''
	self.X_train = self.X[row_idx[:self.N_train],:]
	self.X_test = self.X[row_idx[self.N_train:],:]

	self.y_train = self.y[row_idx[:self.N_train],:]
	self.y_test = self.y[row_idx[self.N_train:],:]
	'''	

	self.W = {} # weights
	self.b = {} # biases
	self.alpha = alpha # learning rate

	self.activation_function, self.activation_gradient = get_activation_function(activation_function)
	self.loss_function = get_cost_function(loss_function)
	self.output_function, self.output_gradient = get_activation_function(output_function)

    def normalization(X):
	P = X.shape[1]
	for p in xrange(P):		
	    mean= np.mean(X[:,p])
	    variance = np.variance(X[:,p])
	    X[:,p]= (X[:,p] - mean)/variance 
	return X	


    def w_initial(self, input, output):
	return sqrt(6.0/(input+output)) 

    def initialize_weights(self):
	input = self.P
	output = self.layers[0]
	w = self.w_initial(input, output)
	self.W[0] = np.random.uniform(-w, w, size=(input, output))
	self.b[0] = 0	

	for i in xrange(1, len(self.layers)-1):
	    input = self.layers[i]
	    output = self.layers[i+1] 
	    w = w_initial(input, output)
	    self.W[i] = np.random.uniform(-w, w, size=(input, output))
	    self.b[i] = 0

    def compute_loss(self, X, y):
	return self.loss_function(X,y)

    def feed_forward(self, X):
	N = X.shape[0]
	self.W_length = self.len(W)	 
	z = {}
	z_new = X

	for layer_idx in xrange(self.W_length):
	    z[layer_idx] = z_new
	    P = self.W[layer_idx].shape[1]
	    z_new = np.zeros((N,P))
	    for p in xrange(P):
		z_new[:, p] = self.activation_function(z[layer_idx].dot(self.W[layer_idx][:,p]) + np.ones(N)*self.b[layer_idx])
	#outer layer
	P = self.W[W_length].shape[1]	
	for p in xrange(P):
	    z_new[p] = self.output_function(z[layer_idx].dot(self.W[W_length][:,p]) + self.b[W_length])
	z[W_length] = z_new
	return z	


    def back_propagation(self, X, y, z):	
	batch_error = self.compute_loss(z[self.W_length], y)

	# update step for output layer
	batch_update = np.zeros(W[self.W_length-1].shape)
	for i,e in enumerate(batch_error):
	    W[self.W_length-1] -= self.alpha*e*self.output_gradient(z[self.W_length-1][i].dot(self.W[self.W_length-1]))*z[self.W_length-1]
	    
	    delta=e*output_gradient(z[self.W_length-1][i].dot(self.W[self.W_length-1]))*z[self.W_length-1]

	# backprop for inside layers
	    for layer_idx in xrange(self.W_length-2, 0, -1):
		delta_old = delta
		delta = self.activation_gradient(z[layer_idx].dot(self.W[layer_idx]))*delta_old.dot(self.W[layer_idx+1])
		self.W[layer_idx] -= self.alpha*delta.dot(self.W[layer_idx-1].dot(z[layer_idx-1])	)
	    delta_old = delta	
	    delta = self.activation_gradient(z[0].dot(self.W[0]))*delta_old.dot(self.W[1])
	    self.W[0] -= self.alpha*delta.dot(z[0])
	    

    def main(self):
	self.initialize_weights()

	z = self.feed_forward(self.X_train)
	self.back_propagation(X, y, z)	



