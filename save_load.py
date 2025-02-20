import numpy as np

def save_model(W_1, b_1, W_2, b_2, filename="model_weights.npz"):
    """
    Saves model 
    """
    np.savez(filename, W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2)
    print(f"Model was saved in {filename}")    
    
    
    
def load_model(filename="model_weights.npz"):
    """
    

    Returns:
    - W_1, b_1, W_2, b_2: parameters of model
    """
    data = np.load(filename)
    W_1 = data["W_1"]
    b_1 = data["b_1"]
    W_2 = data["W_2"]
    b_2 = data["b_2"]
    
    print(f"Model was loaded from {filename}")    
    
    return W_1, b_1, W_2, b_2