"""
TODO: modificare split_normalize_mnist
TODO: dataset preparation
TODO: scrivi mail al prof per riga 82 di main_nn -> assignment 3
"""

import pandas as pd
import numpy as np
import keras as ke
import matplotlib.pyplot as plt
import math

from statistics import mode
from tensorflow.keras.datasets import mnist
from copy import deepcopy

class ML4(ke.Model):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.first = ke.layers.Dense(784, activation="sigmoid")
        self.dense1 = ke.layers.Dense(h, activation="sigmoid")
        self.last = ke.layers.Dense(784, activation="linear")

    def call(self, inputs):
        x = self.first(inputs)
        x = self.dense1(x)
        return self.last(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h})
        return config

def split_normalize_mnist(data, numFolds, currentFold):
    dpf = data.shape[0]/numFolds # data per fold

    idx_0 = int(currentFold*dpf)
    idx_f = int((currentFold+1)*dpf)

    # print(f"{idx_0} -> {idx_f}")

    idx_test = range(idx_0,idx_f)
    idx_train = list(range(idx_0)) + list(range(idx_f,data.shape[0]))

    nc = data.shape[1]
    for i in range(nc):
        col_train = data[idx_train, i]

        minval = col_train.min()
        maxval = col_train.max()

        if maxval == minval:
            data[:, i] = 0.0
        else:
            data[:, i] = (data[:, i] - minval) / (maxval - minval)

    X_train = data[idx_train, :16].astype(float)
    Y_train = data[idx_train, -2:].astype(float)

    X_test = data[idx_test, :16].astype(float)
    Y_test = data[idx_test, -2:].astype(float)

    return [X_train, Y_train, X_test, Y_test]

def isNaN(x):
    if isinstance(x, str):
        return x.lower() == 'nan'
    if isinstance(x, float):
        return math.isnan(x)
    return False

def readData_csv(dataset):
    if dataset == 'data': # puts the output in the last column
        data = read_csv('data.txt', sep='\s+').to_numpy().astype(float)
    else:
        print("Dataset " +str(dataset) + ".txt not in this folder")

    clean_data = [] # data cleaning in case of missing data
    for row in data:
        validRow = True
        for entry in row:
            if isNaN(entry) or entry == '?':
                validRow = False
        if validRow:
            clean_data.append(row)

    clean_data = np.array(clean_data)
    noDe = len(data) - len(clean_data) # number of DELETED entries
    return [clean_data, noDe]

def plotGrid(classes):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    nr = 6
    nc = 6

    N = X_train.shape[0]

    first_idx  = np.where(y_train == classes[0])[0].tolist() # tuple->list
    second_idx = np.where(y_train == classes[1])[0].tolist()

    fig1, ax = plt.subplots(nrows=nr, ncols=nc,layout="tight", figsize=(7,8))
    fig1.tight_layout()
    for i in range(nr):
        for j in range(nc):
            ax[i,j].set_yticks([])
            ax[i,j].set_xticks([])
            ixd = second_idx[5*i+j]
            ax[i,j].imshow(np.reshape(X_train[ixd,:],(28,28)),cmap='Greys')
    plt.show()

def main1():
    classes = (4, 2)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    idx  = np.where(y_train == classes[0] | y_train == classes[1])[0].tolist() # tuple->list

def main2():
    # step 1: dataset preparation
    classes = (4, 2)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    data_idx = np.where(y_train == classes[0] | y_train == classes[1])[0].tolist() # tuple->list
    test_idx = np.where(y_test == classes[0] | y_test == classes[1])[0].tolist()

    # step 2: parameters
    numFolds = 5
    numTrials = 5
    numEpochs = 15
    batchSize = 100
    verb = 0
          
    lperc, hperc = 25, 75   

    k_vec = [12]

    for k in k_vec:
        msevals = []  
        for i in range(numFolds):
            fold_mse_best = None 

            [X_train_i, y_train_i, X_val_i, y_val_i] = split_normalize_mnist(deepcopy(X_data), deepcopy(y_data), numFolds, i)
            
            for trial in range(numTrials):
                model = ML4(k)

                model.name = f"K{k}F{i+1}T{trial+1}" 
                # print(f"k = {k}\t\tFold:{i+1}/{numFolds}\tTrial:{trial+1}/{numTrials}")
                
                model.compile(optimizer = 'adam', loss = ke.losses.MeanSquaredError())
                model.fit(X_train_i, y_train_i, verbose = verb, batch_size = batchSize, validation_split = .1, epochs = numEpochs)
                
                # EVALUATION
                Y_pred = model.predict(X_val_i, verbose = verb)
                mse = ((Y_pred - y_val_i)**2).sum()/(len(Y_pred)*2)
                
                # Update fold best
                if fold_mse_best is None or mse < fold_mse_best:
                    fold_mse_best = mse
                
                # Reset model
                del model

            msevals.append(fold_mse_best)

        msevals = np.array(msevals, dtype=float)
        
        # Print results
        if len(msevals)>0:
            print(f"--- Result Summary for k = {k} ---")
            print(msevals) 
            low, med, high = np.percentile(msevals, (lperc, 50, hperc))
            print(f"mse = {med:.3F} (typical)\naccuracy in [{low:.3F}, {high:.3F}] with probability >= {(hperc-lperc)/100:.2F}\n")
            
def main3():
    runs = 1 # numero di test

    test_size = 200      # max 10000
    train_size = 5000    # max 60000

    k_vector = [10] #[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

    accuracies = []
    stds = []

    accuracy_i = []

    (T_train, Y_train), (T_test, Y_test) = mnist.load_data()

    for k in k_vector:
        accuracy_i = []
        for _ in range(runs):
            # Shuffle training data
            idx = np.arange(60000)
            np.random.shuffle(idx)
            T_train_i = T_train[idx]
            Y_train_i = Y_train[idx]

            # Shuffle test data
            idx = np.arange(10000)
            np.random.shuffle(idx)
            T_test_i = T_test[idx]
            Y_test_i = Y_test[idx]

            # Crop
            T_train_i = T_train_i[:train_size]
            Y_train_i = Y_train_i[:train_size]
            T_test_i  = T_test_i[:test_size]
            Y_test_i  = Y_test_i[:test_size]

            # Normalizzo train
            T_train_i = T_train_i.astype(np.float32) / 255.0
            T_test_i  = T_test_i.astype(np.float32) / 255.0
            Y_train_i = Y_train_i.astype(np.int32)
            Y_test_i  = Y_test_i.astype(np.int32)

            kNN = classifier.kNN(k)
            kNN.fit(T_train_i, Y_train_i)

            acc = kNN.test(T_test_i, Y_test_i)   # returns vector
            accuracy_i.append(acc * 100)

        accuracies.append(np.mean(accuracy_i))
        stds.append(np.std(accuracy_i))

    # --- Risultati  ---
    print("Result summary for {mnist}\nnumber of runs:\t"+str(runs)+"\ntrain size:\t"+str(train_size)+"\ntest size:\t"+str(test_size))
    print("k\taccuracy\tstd_dev")
    print("———————————————————————————————")
    for k, acc, std in zip(k_vector, accuracies, stds):
        print(f"{k}\t{acc:.3f}\t\t{std:.3f}")

if __name__ == '__main__':
    main2()