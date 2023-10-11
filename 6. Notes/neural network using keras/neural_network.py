# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:11:41 2021
https://deeplizard.com/learn/video/RznKVRTFkBY
@author: Prakhar Sharma
"""

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %% Data preparation and pre-processing

train_samples=[]
train_labels=[]

'''
Example data: An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial.
The trial had 2100 paritcipants. Half were under 65 yeras old, half were 65 years or older.
Around 95% of patients 65 or older experienced side effects.
Around 95% of patients under 65 experienced no side effects.
'''

for i in range(50): # generating outliers
    # The ~5% of younger individuals who did experience side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    # The ~5% of older individuals who did not experience side effects
    random_older=randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000): # generating main dataset
    # The ~95% of younger individuals who did not experience side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    # The ~95% of older individuals who did experience side effects
    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
    
# Plotting the data
# plt.figure(1)
# plt.plot(train_samples,train_labels,'o')

# preprocessing
train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
train_samples,train_labels=shuffle(train_samples,train_labels) # always shuffle the data if the order doesn't matters 

# resampling and rescaling within 0 and 1
scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_samples=scaler.fit_transform(train_samples.reshape(-1,1))

# %% Keras sequential model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model=Sequential([
    Dense(units=16, input_shape=(1,),activation='relu'), # first hidden layer
    Dense(units=32,activation='relu'),
    Dense(units=2,activation='softmax')    # output layer 
    ])

# Here relu has a range of (0,inf) and softmax (0,1) so softwmax
# is used in last layer to to obtain a probability type output between (0,1)


# To create a visual summary of our NN
model.summary()

# prepare the model for training
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the NN
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)
# verbose is used to set the level of the output

# %% Validation
# To see the general performance of out NN and see if there is overfitting
# if the val_accuracy is low then we have overfitting
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

# %% Testing the data for prediction

test_samples=[]
test_labels=[]

for i in range(10): # generating outliers
    # The ~5% of younger individuals who did experience side effects
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    # The ~5% of older individuals who did not experience side effects
    random_older=randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200): # generating main dataset
    # The ~95% of younger individuals who did not experience side effects
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    # The ~95% of older individuals who did experience side effects
    random_older= randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

# preprocessing
test_labels=np.array(test_labels)
test_samples=np.array(test_samples)
test_samples,test_labels=shuffle(test_samples,test_labels) 


scaled_test_samples=scaler.fit_transform(test_samples.reshape(-1,1))
 
# prediction
predict=model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
# see the output: it is a probability of side effect or not for this problem

# Getting index of the highest probability for each row

rounded_predictions= np.argmax(predict, axis=-1)

# %% Confusion matrix
# https://deeplizard.com/learn/video/km7pxKy4UHU

from sklearn.metrics import confusion_matrix
import itertools

cm=confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
# This function is copied directly from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels=['No_side_effects','had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# %% Save and load a model

import os.path
if os.path.isfile('saved_model/model.h5') is False:
    model.save('saved_model/model.h5')
    
# This saves the model, state of the optimizer, weights and biases etc.

# Load a model
from tensorflow.keras.models import load_model
new_model= load_model('saved_model/model.h5')

new_model.summary()
# Obtaining weights and bias
# Method 1 
new_model.get_weights()

# Method 2:
# index 1 is the weights array and index2 is the bias array
W_Input_Hidden = new_model.layers[0].get_weights()
W_Output_Hidden = new_model.layers[1].get_weights()

# Obtaining the optimizer used
new_model.optimizer

# Saving the model as json file: save the architecture only
json_string=model.to_json()

# saving as a YAML file
yaml_string=model.to_yaml()

# see outputs for botht he strings that saves the architecture

# Model reconstruction from the JSON file:
from tensorflow.keras.models import model_from_json
model_architecture= model_from_json(json_string)

model_architecture.summary()

# %% Plotting the history


history=model.fit(x=scaled_train_samples, y=train_labels,validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)
history_data=history.history
history_keys=history.history.keys()
#  "Accuracy"
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()




 








