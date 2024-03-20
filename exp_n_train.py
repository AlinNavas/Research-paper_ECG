# %%
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.metrics import *

# %%
from sklearn.metrics import confusion_matrix, f1_score


# %%


length = 277
DS1_data=pd.read_csv('./data/train_test/DS1_data.csv',header=None).values
DS2_data=pd.read_csv('./data/train_test/DS2_data.csv',header=None).values



# %%
train_patients = DS1_data
test_patients = DS2_data


from sklearn.utils import resample

N = train_patients[train_patients[:,-2]==1.0]
L = train_patients[train_patients[:,-2]==2.0]
R = train_patients[train_patients[:,-2]==3.0]
V = train_patients[train_patients[:,-2]==4.0]
A = train_patients[train_patients[:,-2]==5.0]
F = train_patients[train_patients[:,-2]==6.0]


seed=26
np.random.seed(seed)
def downsample(arr, n, seed):
    downsampled = resample(arr,replace=False,n_samples=n, random_state=seed)
    return downsampled
  
def upsample(arr, n, seed):
    upsampled = resample(arr,replace=True,n_samples=n,random_state=seed)
    return upsampled

all_class = [N,L,R,V,A,F]
abn_class = [L,R,V,A,F]

# %%
N.shape, L.shape, R.shape, V.shape, A.shape, F.shape,

# %%

mean_val = np.mean([len(i) for i in abn_class], dtype= int)
train_sampled = []

for i in abn_class:
    if i.shape[0]> mean_val:
        i = downsample(i,mean_val,seed)
    elif i.shape[0]< mean_val:
        i = upsample(i, mean_val,seed)
    train_sampled.append(i)
    
train_sampled = np.concatenate(train_sampled)



# %%
f1_array = []
for x in np.arange(1, 2.1, 0.5):   
    downsampled_data = downsample(all_class[0],int(x * mean_val),seed)
    train_sample = np.concatenate((train_sampled, downsampled_data), axis=0)
    np.random.shuffle(train_sample)
    train_sampled_all = train_sample

    train_values = train_sampled_all
    test_values = pd.read_csv('./data/train_test/test_patients.csv').values

    # Separate the training and testing data, and one-hot encode Y:
    X_train = train_values[:,:-2]
    X_test = test_values[:,:-2]
    y_train = train_values[:,-2]
    y_test = test_values[:,-2]
    X_test1 = X_test.reshape(-1, X_train.shape[1], 1)
    y_test1=to_categorical(y_test)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,stratify=y_train)

    X_train1 = X_train.reshape(-1, X_train.shape[1], 1)
    y_train1=to_categorical(y_train)

    X_val1 = X_val.reshape(-1, X_train.shape[1], 1)
    y_val1=to_categorical(y_val)


    def mlpmodel():
        mlpmodel = Sequential()
        mlpmodel.add(Dense(275, input_shape=(X_train.shape[1],)))
        mlpmodel.add(Dense(50))
        mlpmodel.add(BatchNormalization())
        mlpmodel.add(Activation('relu'))
        mlpmodel.add(Dense(150))
        mlpmodel.add(BatchNormalization())
        mlpmodel.add(Activation('relu'))
        mlpmodel.add(Dense(900))
        mlpmodel.add(BatchNormalization())
        mlpmodel.add(Activation('relu'))
        mlpmodel.add(Dense(400))
        mlpmodel.add(BatchNormalization())
        mlpmodel.add(Activation('relu'))
        mlpmodel.add(Dense(7, activation='softmax'))
        
        return mlpmodel

    mlp = mlpmodel()
    

    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min',min_delta=0.001, restore_best_weights=True)
    history = mlp.fit(X_train, y_train1, validation_data=(X_val, y_val1), epochs=50,callbacks=[early_stopping], verbose=1)
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
    print("F1 Score for value of " + str(x) + ":", f1)

    f1_array.append((f1, x))
print(f1_array)

# %% [markdown]
# 
# Nm = train_sampled_all[train_sampled_all[:,-2]==1.0]
# Lm = train_sampled_all[train_sampled_all[:,-2]==2.0]
# Rm = train_sampled_all[train_sampled_all[:,-2]==3.0]
# Vm = train_sampled_all[train_sampled_all[:,-2]==4.0]
# Am = train_sampled_all[train_sampled_all[:,-2]==5.0]
# Fm = train_sampled_all[train_sampled_all[:,-2]==6.0]

# %% [markdown]
# Nm.shape, Lm.shape, Rm.shape, Vm.shape, Am.shape, Fm.shape, train_sampled_all.shape

# %% [markdown]
# 
# with open('./data/train_test/train_patients.csv', "wb") as fin:
#     np.savetxt(fin, train_sampled_all, delimiter=",", fmt='%f')
# 
# with open('./data/train_test/test_patients.csv', "wb") as fin:
#     np.savetxt(fin, test_patients, delimiter=",", fmt='%f')


