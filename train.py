import numpy as np
from model import forward_propagation, update_parameters, cross_entropy_loss

def train_neural_network(alpha, batch_size, Number_of_epochs, X_train, Y_train):
    
    # Parameters of model initialization: 
    
    mean = 0
    std = 1
    shape_W_1 = (128,784)
    shape_W_2 = (10,128)
    shape_b_1 = (128,1)
    shape_b_2 = (10,1)
    
    W_1 = np.random.normal(loc=mean, scale=std, size=shape_W_1) * np.sqrt(1/784) 
    W_2 = np.random.normal(loc=mean, scale=std, size=shape_W_2) * np.sqrt(1/128)

    b_1 = np.zeros(shape_b_1) 
    b_2 = np.zeros(shape_b_2) 
    
    
    patience = 5  # Number of epochs without any changes after which we stop
    best_loss = float("inf")
    patience_counter = 0

    loss_history = []

    for epoch in range(Number_of_epochs):
        
        Number_of_train_examples = X_train.shape[0]
        indices = list(range(Number_of_train_examples))         # mixing of indicies
        
        np.random.shuffle(indices)
        
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        Number_of_batches = Number_of_train_examples//batch_size
        

        
        for i in range(Number_of_batches):
            
            indices_i = indices[i*batch_size : (i+1)*batch_size]
            batch_X = X_train_shuffled[indices_i]
            batch_Y = Y_train_shuffled[indices_i]
            
            W_1, b_1, W_2, b_2 = update_parameters(alpha = alpha , X = batch_X, Y = batch_Y, W_1 = W_1, W_2 = W_2, b_1 = b_1, b_2 = b_2)
        
        # Loss function calculations after each of the epochs:
        
        _, _, _, A_2_train = forward_propagation(X_train, W_1, W_2, b_1, b_2)
        loss = cross_entropy_loss(Y_train, A_2_train, W_1, W_2)  
        loss_history.append(loss)
            
        # Statistics printing:
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{Number_of_epochs}, Loss_function: {loss:.4f}")
        
        # Here we check how loss function does behave:
        
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0  
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stop on {epoch+1}'s epoch")
            break  

    print("Training is finished")
        
    return W_1, b_1, W_2, b_2, loss_history