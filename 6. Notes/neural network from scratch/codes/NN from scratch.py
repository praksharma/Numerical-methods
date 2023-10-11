# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:28:24 2021

@author: Prakhar Sharma
"""

import numpy as np
# Initialize our parameters to matrices of small, random values of the correct shape
def init_params():
    # 10 the number of digits that MNISt dataset contains and 784 is the number of pixels in each image
    W1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1, b1, W2, b2

#a,b,c,d=init_params()

# forward propagation

def forward_prop(W1, b1, W2, b2, X):
    Z1=W1.dot(X)+b1 # output from the neuron
    A1=ReLu(Z1)     # activated function
    Z2=W2.dot(A1)+b2
    A2=ReLu(Z2)  
    return Z1, A1, Z2, A2

def ReLu(Z):
    return np.maximum(Z,0)

def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

# Notice that we returned not just our final predictions but also Z1, A2, and Z2 from our forward prop function. This is because we’ll now pass them into a backprop function:


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y=one_hot(Y) # creating an empty one hot vector
    # computing partial derivatives
    # Second hidden layer 
    dZ2=A2-one_hot_Y
    dW2=1/m*dZ2.dot(A1.T)
    db2=1/m*np.sum(dZ2)
    # first hidden layer
    dZ1=W2.T.dot(dZ2)*ReLu_deriv(Z1)
    dW1=1/m*dZ1.dot(X.T)
    db1=1/m*np.sum(dZ1)
    return dW1, db1, dW2, db2
def ReLu_deriv(Z):
    # Returns a boolean of 0s and 1s. i.e the derivative of ReLu
    return Z>0 

def one_hot(Y):
    one_hot_Y=np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1
    one_hot_Y=one_hot_Y.T
    return one_hot_Y

# Updating the parameters using gradient descent
def update_params(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2):
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    return W1, b1, W2, b2

# Getting the indices of the maximum value
def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size


# Tieing all these functions in a wrapper function to just call them

def gradient_descent(X,Y,alpha, iterations):
    '''
    X,Y is our training data
    alpha is our selected learning rate
    iterations is the number of epochs
    '''
    # Initilize parameters
    W1, b1, W2, b2 = init_params()
    # Epochs
    for i in range(iterations):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        # Backpropagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        # Updating params
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2)
        
        # Printing the preditions and accracy for some of the iterations
        if i%10 == 0:
            print('Epoch: ',i)
            predictions=get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2  # returns final weights and biases
 
# Following 2 functions are to test the predictions for each data based on the trained weights and biases
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title('Label: ['+str(label)+']   Prediction: '+str(prediction))
    plt.show()

# %% inputting the data in the Neurral network

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# Training the parameters (alpha =0.10)
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Testin our model using the trained parameters
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)


    
    

