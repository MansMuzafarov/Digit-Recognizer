# MNIST Neural Network from Scratch (NumPy-Only Implementation)

## Overview

This project presents a **fully manual implementation** of a **Multi-Layer Perceptron (MLP)** for digit recognition using the **MNIST dataset**. The **entire neural network architecture and optimization algorithms** are implemented **exclusively using NumPy**.

### Key Features:

- **Forward and backward propagation implemented manually**
- **Gradient descent optimization (mini-batch approach)**
- **Cross-entropy loss function with L2 regularization**
- **ReLU and Softmax activation functions**
- **Model evaluation using accuracy and confusion matrix**
- **Weights saving/loading functionality**

📌 **Note:**

- **TensorFlow** is used **only for loading the MNIST dataset**.
- **Scikit-learn** is used **only for confusion matrix visualization**.
- **No pre-built deep learning frameworks (e.g., PyTorch, Keras, TensorFlow) were used for model training.**

---

## Model Architecture

- **Input Layer:** 784 neurons (corresponding to 28×28 grayscale images)
- **Hidden Layer:** 128 neurons (ReLU activation)
- **Output Layer:** 10 neurons (Softmax activation for classification)
- **Optimization:** Mini-batch gradient descent
- **Loss function:** Cross-entropy with L2 regularization

---

## Installation

1. **Clone the repository:**

   git clone https://github.com/MansMuzafarov/Digit-Recognizer---Neural-Network-solution  
   cd Digit-Recognizer---Neural-Network-solution
   
2. **Install dependencies:**
   
   pip install -r requirements.txt
   



## Running the Project (`main.py`)

### **Main Execution 

The `main.py` script **either trains or loads the neural network**, evaluates it, and visualizes results.

### Execution Workflow:

1. Loads and **preprocesses** the MNIST dataset.
2. **Trains the model** if `train_model = True`, or **loads a pre-trained model** otherwise.
3. **Plots loss function history** (if training is performed).
4. **Evaluates accuracy** on test data.
5. **Generates a confusion matrix**.
6. **Displays sample predictions** from the test set.

### Execution Command:

```sh
python main.py
```

### Training vs. Preloading Model

Modify `main.py`:

```python
train_model = False  # Set to True to train the model from scratch
```

- `False`: Loads the **pre-trained** model (`trained_model.npz`).
- `True`: **Trains a new model** and saves weights.

---


## Project Structure


Digit-Recognizer---Neural-Network-solution/  

├── data_preprocessing.py  # Data loading and normalization (MNIST)  
├── model.py               # Neural network implementation (forward/backprop/loss-function)  
├── train.py               # Training procedure (mini-batch gradient descent)  
├── evaluation.py          # Accuracy calculation, confusion matrix visualization  
├── save_load.py           # Model weight saving & loading functions  
├── config.py              # Hyperparameters (learning rate, batch size, epochs)  
├── utils.py               # Activation functions (ReLU, Softmax, etc.)  
├── requirements.txt       # Required dependencies  
├── main.py                # Main script for training and evaluation  
├── README.md              # Project documentation (this file)  
└── .gitignore             # Ignored files (cache, temp files)  




## Hyperparameter Configuration

All **training settings** are defined in `config.py`:

```python
alpha = 0.01  # Learning rate
number_of_epochs = 100  # Number of training epochs
batch_size = 32  # Batch size for mini-batch gradient descent
```

These values can be modified within `config.py` or overridden using command-line arguments.

---

## License

MIT License - you are free to modify and use this project.

