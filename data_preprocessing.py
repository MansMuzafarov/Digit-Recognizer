import numpy as np
import tensorflow as tf



def one_hot_encoding(y, number_of_numbers = 10):
    
    N_samples = y.shape[0]
    
    one_hot_matrix = np.zeros((N_samples, number_of_numbers))
    
    for i in range(N_samples):
        
        label = y[i]                                # label of ith example    
        one_hot_matrix[i,label] = 1
        
    return one_hot_matrix


def data_preprocessing():
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()   

    Num_of_images_in_train_data = X_train.shape[0]
    Num_of_images_in_test_data = X_test.shape[0]
    size_of_the_image = X_train.shape[1]     #Square image (equal length and width)



    X_train = X_train.reshape(Num_of_images_in_train_data, size_of_the_image*size_of_the_image)  

    X_test = X_test.reshape(Num_of_images_in_test_data, size_of_the_image*size_of_the_image)


    # Normalization:

    X_train = X_train.astype(np.float32) / 255.0

    X_test = X_test.astype(np.float32) / 255.0


    # y_train and y-test to one-hot format:

    Y_train = one_hot_encoding(y = y_train)

    Y_test = one_hot_encoding(y = y_test)
    
    return X_train, Y_train, X_test, Y_test