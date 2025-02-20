import numpy as np

def ReLU(X): # calculates ReLU for each component of the input matrix X
    
    return np.maximum(0,X)

def ReLU_derrivative(X):
    
    return (X>0).astype(float)



def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)