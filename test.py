import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
import os
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import *
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
sns.set()
import warnings
warnings.filterwarnings("ignore")
length = 277
# Load the training and testing data:
train_values = np.empty(shape=[0, length])
test_values = np.empty(shape=[0, length])

train_beats = glob.glob('./data/train_test/train_beats.csv')
test_beats = glob.glob('./data/train_test/test_beats.csv')

for j in train_beats:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    train_values = np.append(train_values, csvrows, axis=0)

for j in test_beats:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    test_values = np.append(test_values, csvrows, axis=0)
    
print(train_values.shape)
print(test_values.shape)

# Separate the training and testing data, and one-hot encode Y:
X_train = train_values[:,:-2]
X_test = test_values[:,:-2]
y_train = train_values[:,-2]
y_test = test_values[:,-2]
X_train1 = X_train.reshape(-1, X_train.shape[1], 1)
X_test1 = X_test.reshape(-1, X_train.shape[1], 1)
y_train1=to_categorical(y_train)
y_test1=to_categorical(y_test)

# Combine everything again:
X = np.concatenate((X_train1, X_test1), axis = 0)
Y = np.concatenate((y_train1, y_test1), axis = 0)

from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv1D, MaxPooling1D

def getModel():
    
    cnnmodel = Sequential()
    cnnmodel.add(Conv1D(filters=128, kernel_size=16,padding='same', activation='relu',input_shape=(X_train1.shape[1],X_train1.shape[2])))
    cnnmodel.add(BatchNormalization())
    cnnmodel.add(Conv1D(filters=32, kernel_size=16,padding='same', activation='relu'))
    cnnmodel.add(BatchNormalization())
    cnnmodel.add(Conv1D(filters=9, kernel_size=16,padding='same', activation='relu'))
    cnnmodel.add(MaxPooling1D(pool_size=4,padding='same'))
    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(256, activation='relu'))
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dense(32, activation='relu'))
    cnnmodel.add(Dense(9, activation='softmax'))
    cnnmodel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return cnnmodel

cnnmodel = getModel()
cnnmodel.summary()


# Get all the data that corresponds to one single class (class 8 / 'I'):
for class_num in range(1, 9):

        N_test = test_values[test_values[:,-2]==class_num]

        Xs_test = N_test[:,:-2]
        ys_test = N_test[:len(N_test),-2]

        Xs_test1 = X_test.reshape(-1, X_test.shape[1], 1)
        ys_test1 = to_categorical(y_test)

        # Permutation feature importance:
        # Build the machine learning model with the training data:
        M = np.zeros((Xs_test1.shape[0], 11))

        cnnmodel = getModel()
        cnnmodel.fit(X_train1, y_train1, verbose = 0, epochs = 10, validation_split = 0.2, batch_size = 64)
                
        # Predict on the test fold without permutations:
        pred_k = cnnmodel.predict(Xs_test1)     
                
                # For every feature:
        for slice_start in range(0, 275, 25):
                # Permute and predict:
                x_permuted = np.copy(Xs_test1)
                x_slice = Xs_test1[:, slice_start:slice_start+25]
                x_slice_permuted = np.random.permutation(x_slice)
                x_permuted[:, slice_start:slice_start + 25] = x_slice_permuted
                pred_perm = cnnmodel.predict(x_permuted)
                
        # Compute importance:
                importance = ((np.argmax(ys_test1, axis = 1) - np.argmax(pred_perm, axis = 1))**2 
                                        - (np.argmax(ys_test1, axis = 1) - np.argmax(pred_k, axis = 1))**2)
                M[:, slice_start // 25] = importance    
        mean_importance = np.mean(M, axis = 0)


        indices_sort = np.argsort(-1 * mean_importance)
        slices = np.arange(1, 12)

        fig, ax = plt.subplots(1, 2, figsize = (15, 4))
        ax[0].bar(range(11), mean_importance[indices_sort])
        ax[0].set_title('Feature importance per segment for the CNN model')
        ax[0].set_xticks(np.arange(11))
        ax[0].set_xticklabels(slices[indices_sort].astype(int))
        ax[0].set_xlabel('Slice')
        ax[0].set_ylabel('Feature importance')

        ecg_normalized = (Xs_test[20, :] - Xs_test[20, :].min()) / (Xs_test[20, :].max() - Xs_test[20, :].min())
        feature_importance_normalized = (mean_importance - mean_importance.min()) / (mean_importance.max() - mean_importance.min())
        ax[1].plot(np.arange(len(ecg_normalized)), ecg_normalized, label='ECG data')
        ax[1].plot(np.repeat(feature_importance_normalized, 25), label = 'Feature importance')
        ax[1].set_title('Feature importance per segment \nfor the CNN model on an avg of'+ str(class_num)+ 'ECG')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('ECG signal / feature importance')
        ax[1].legend()
        image_name = 'cnn_permutation_feature_importance_results' + str(class_num) + '.jpg'
        fig.savefig(image_name, dpi = 400)