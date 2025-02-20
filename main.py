import os
import matplotlib.pyplot as plt
from data_preprocessing import data_preprocessing
from train import train_neural_network
from save_load import save_model, load_model
from evaluation import find_accuracy_of_the_model, visualize_n_predictions, plot_confusion_matrix
from config import train_model, alpha, number_of_epochs, batch_size


# ====== Data loading and preprocessing ======
X_train, Y_train, X_test, Y_test = data_preprocessing()


model_path = "trained_model.npz"

# ====== Model training or loading ======
if train_model or not os.path.exists(model_path):
    print("\n🔵 Training the model...\n")
    W_1, b_1, W_2, b_2, loss_history = train_neural_network(
        alpha=alpha, batch_size=batch_size, Number_of_epochs=number_of_epochs, X_train=X_train, Y_train=Y_train
    )
    save_model(W_1, b_1, W_2, b_2, model_path)
else:
    print("\n🟢 Loading pre-trained model...\n")
    W_1, b_1, W_2, b_2 = load_model(model_path)
    loss_history = []  # No loss history if loading a pre-trained model

# ====== Plot loss function history (only if trained) ======
if train_model and loss_history:
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value")
    plt.title("Loss Function History")
    plt.savefig("Loss_function_history.png")
    plt.close()

# ====== Model evaluation ======
accuracy = find_accuracy_of_the_model(X_test, Y_test, W_1, W_2, b_1, b_2)


# ====== Visualizing Results ======
plot_confusion_matrix(X_test, Y_test, W_1, W_2, b_1, b_2)

# Direct examples:
n = 5
visualize_n_predictions(X_test, Y_test, W_1, W_2, b_1, b_2, n=n)
