import numpy as np
from numpy.linalg import norm

def squared_loss(x,y):
    return 0.5*norm(x-y)**2

def get_cost_function(func_name):
    if func_name == 'squared loss':
	return squared_loss




