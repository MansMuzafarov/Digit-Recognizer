import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from model import forward_propagation
from sklearn.metrics import confusion_matrix



def predict(X, W_1, W_2, b_1, b_2, return_probabilities = False):
    
    Z_1, A_1, Z_2, A_2 = forward_propagation(X, W_1, W_2, b_1, b_2)
    
    predicted_numbers = np.argmax(A_2, axis = 1)
    if return_probabilities == True:
    
     return predicted_numbers , A_2
    
    return predicted_numbers

def find_accuracy_of_the_model(X_test, Y_test, W_1, W_2, b_1, b_2):
    """
    Finds accuracy of the model
    
    W_1, W_2, b_1, b_2 - parameters (from gradient descent)
    
    """
    # Make predictions:
    predictions = predict(X_test, W_1, W_2, b_1, b_2)
    
    # True labels:
    true_labels = np.argmax(Y_test, axis=1)
    
    # accuracy calculation:
    accuracy = np.mean(predictions == true_labels)
    
    print(f"Accuracy of the model: {accuracy * 100:.2f}%")
    
    return accuracy




def visualize_n_predictions(X_test, Y_test, W_1, W_2, b_1, b_2, n):
    """
    Visualizes n random examples from test data with neural network predictions.
    
    """
    indices = random.sample(range(X_test.shape[0]), n)  
    
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))  
    
    for i, ax in enumerate(axes):
        index = indices[i]
        
        
        sample_X = X_test[index].reshape(1, -1)  # to (1, 784)
        sample_Y = np.argmax(Y_test[index])  # True label
        
        # Make predictions:
        predicted_label, probabilities = predict(sample_X, W_1, W_2, b_1, b_2, return_probabilities=True)

        # Drawing:
        ax.imshow(sample_X.reshape(28, 28), cmap="gray")
        ax.set_title(f"True: {sample_Y}\nPrediction: {predicted_label[0]}\nConfidence: {probabilities[0, predicted_label[0]]:.2f}")
        ax.axis("off")
    
    plt.savefig('Predictions.png')
    plt.show()



def plot_confusion_matrix(X_test, Y_test, W_1, W_2, b_1, b_2):
    true_labels = np.argmax(Y_test, axis=1)
    predictions = predict(X_test, W_1, W_2, b_1, b_2)

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion_Matrix.png")
    plt.close()