import matplotlib.pyplot as plt
import os
import pandas as pd
from save_load import save_model, load_model
from data_preprocessing import data_preprocessing_kaggle
from evaluation import predict
from config import train_model, alpha, number_of_epochs, batch_size
from train import train_neural_network



def make_predictions_for_kaggle():
    
    
    X_train, Y_train, X_test, test_ids = data_preprocessing_kaggle()
    
    

    model_path = "trained_model_kaggle.npz"

    # ====== Model training or loading ======
    if train_model or not os.path.exists(model_path):
        print("\n🔵 Training the model... (Kaggle)\n")
        W_1, b_1, W_2, b_2, loss_history = train_neural_network(
            alpha=alpha, batch_size=batch_size, Number_of_epochs=number_of_epochs, X_train=X_train, Y_train=Y_train
        )
        save_model(W_1, b_1, W_2, b_2, model_path)
    else:
        print("\n🟢 Loading pre-trained model... (Kaggle)\n")
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

    # Kaggle predictions:

    predictions = predict(X_test, W_1, W_2, b_1, b_2)
    
    df_submission = pd.DataFrame({"ImageId": test_ids, "Label": predictions})
    df_submission.to_csv("submission.csv", index=False)
    print("Predictions were saved to submission.csv file!")

    


