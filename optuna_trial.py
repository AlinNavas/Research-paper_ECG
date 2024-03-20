# %%
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
from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.metrics import *
sns.set()
import warnings
warnings.filterwarnings("ignore")
length = 277
!pip install optuna
!pip install optuna-integration

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
train_values = pd.read_csv('./data/train_test/train_patients.csv').values
test_values = pd.read_csv('./data/train_test/test_patients.csv').values

print(train_values.shape)
print(test_values.shape)


# Separate the training and testing data, and one-hot encode Y:
X_train = train_values[:,:-2]
X_test = test_values[:,:-2]
y_train = train_values[:,-2]
y_test = test_values[:,-2]
X_test1 = X_test.reshape(-1, X_train.shape[1], 1)
y_test1=to_categorical(y_test)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,stratify=y_train)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
X_train1 = X_train.reshape(-1, X_train.shape[1], 1)
y_train1=to_categorical(y_train)

X_val1 = X_val.reshape(-1, X_train.shape[1], 1)
y_val1=to_categorical(y_val)


# %%
import optuna
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def objective(trial):
    # Define the search space for hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 5)
    layer_size = trial.suggest_int('layer_size', 50, 1000)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    activation = trial.suggest_categorical('activation', ['relu','tanh'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Create the MLP model with hyperparameters
    mlpmodel = Sequential()
    mlpmodel.add(Dense(layer_size, input_shape=(X_train.shape[1],)))
    mlpmodel.add(Dropout(dropout_rate))
    
    for _ in range(num_layers):
        mlpmodel.add(Dense(layer_size))
        if use_batch_norm:
            mlpmodel.add(BatchNormalization())
        mlpmodel.add(Activation(activation))
        mlpmodel.add(Dropout(dropout_rate))
    
    mlpmodel.add(Dense(7, activation='softmax'))

    # Compile the model with hyperparameters
    mlpmodel.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model with hyperparameters
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.001, restore_best_weights=True)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    history = mlpmodel.fit(X_train, y_train1, validation_data=(X_val, y_val1), epochs=50, callbacks=[early_stopping, pruning_callback], verbose=1)
    
    # Return the validation accuracy as the objective value
    return history.history['val_accuracy'][-1]

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and objective value
best_params = study.best_params
best_value = study.best_value
print('Best Hyperparameters:', best_params)
print('Best Objective Value:', best_value)


# %%
class_counts = np.bincount(y_test.astype(int))
print(class_counts)


# %%
from sklearn.metrics import confusion_matrix, f1_score

# Predict the test data
y_pred = mlp.predict(X_test1)

# Convert the predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(cm)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred_labels, average='weighted')
print("F1 Score:", f1)




# %%


# %%



