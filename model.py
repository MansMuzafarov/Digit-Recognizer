import numpy as np
from utils import ReLU, ReLU_derrivative, softmax


def forward_propagation(X, W_1, W_2, b_1, b_2):   # X - batch of m examples in a matrix form of (m,784) 
    
    # 1st step:
    Z_1 = np.dot(X, W_1.T) + b_1.T
    A_1 = ReLU(Z_1)
    # 2nd step (output):
    Z_2 = np.dot(A_1, W_2.T) + b_2.T
    A_2 = softmax(Z_2)
    
    return Z_1, A_1, Z_2, A_2



def backpropagation(X, Y, W_1, W_2, Z_1, A_1, A_2, lambda_coef = 0.01):
    
    m = X.shape[0]    # number of examples in the batch
    
    # X - matrix of input data (batch of m examples) => matrix of the form (m,784)
    # Y - matrix of true values in one-hot format for each examples => matrix of the form (m,10)
    # Z_1, A_1, Z_2, A_2 - matricies from forward propagation of the neural network 
    # Z_1, A_1 of the shape - (m,128)
    # Z_2, A_2 of the shape - (m, 10)
    # W_1 - matrix of weights in the hidden layer - (128,784)
    # W_2 - matrix of weights in the output layer (10,128)
    
    
    # Last layer gradients:
    dZ_2 = A_2 - Y                                     # (m,10)
    
    dW_2 = (1/m) * np.dot(dZ_2.T, A_1) + (lambda_coef / m) * W_2                 # (10,128)
    
    db_2 = (1/m) * np.sum(dZ_2, axis = 0, keepdims = True)                      # (10,1)
    
    # W-1 hidden layer gradients: 
    
    dZ_1 = np.dot(dZ_2, W_2) * ReLU_derrivative(Z_1)   # (m,128)
    
    dW_1 = (1/m) *np.dot(dZ_1.T, X) + (lambda_coef/ m) * W_1                           # (128,784)
    
    db_1 = (1/m)*np.sum(dZ_1, axis = 0, keepdims = True)                      # (128,1)
    
    return dW_2, db_2, dW_1, db_1


def update_parameters(alpha, X, Y, W_1, W_2, b_1, b_2 ):
    
    
    Z_1, A_1, Z_2, A_2 = forward_propagation(X, W_1, W_2, b_1, b_2)
    
    
    
    dW_2, db_2, dW_1, db_1 = backpropagation(X = X, Y = Y, W_1 = W_1, W_2 = W_2,  Z_1 = Z_1, A_1 = A_1, A_2 = A_2 )
    
    # Parameters update:
   
    W_2 -= alpha * dW_2
    b_2 -= alpha * db_2.T   
    W_1 -= alpha * dW_1
    b_1 -= alpha * db_1.T
    
    return W_1, b_1, W_2, b_2


# Loss-function: (cross-entropy)


def cross_entropy_loss(y_true, y_pred, W_1, W_2, lambda_=0.01, eps=1e-8):
 
    m = y_true.shape[0]
    
    # Standard cross-entropy
    loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    # Regularization term
    l2_regularization = (lambda_ / (2 * m)) * (np.sum(W_1**2) + np.sum(W_2**2))

    return loss + l2_regularization