"""
TODO: scrivi mail al prof per riga 82 di main_nn -> assignment 3
"""

import numpy as np
import neuralNetworks as nn
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from tensorflow.keras.datasets import mnist
from copy import deepcopy

def kFold_mnist(X_data, y_data, numFolds, currentFold):
    dpf = X_data.shape[0]/numFolds # data per fold

    idx_0 = int(currentFold*dpf)
    idx_f = int((currentFold+1)*dpf)

    idx_validation = range(idx_0,idx_f)
    idx_train = list(range(idx_0)) + list(range(idx_f,X_data.shape[0]))

    X_train = X_data[idx_train].astype(float)
    y_train = y_data[idx_train].astype(float)

    X_validation = X_data[idx_validation].astype(float)
    y_validation = y_data[idx_validation].astype(float)

    return [X_train, y_train, X_validation, y_validation]

def getPerformances(X_data, y_data, numFolds, numTrials, batchSize, numEpochs, verb, lperc, hperc, k):
    msevals = []  
    for i in range(numFolds):
        fold_mse_best = float('inf') 

        # Ensure kFold_mnist returns X_val_i as images for reconstruction tasks
        [X_train_i, _ , X_val_i, _] = kFold_mnist(deepcopy(X_data), deepcopy(y_data), numFolds, i)

        for trial in range(numTrials):
            tf.keras.backend.clear_session() # Clears the graph for the new trial
            
            model = nn.ML4(k)
            print(f"-- k={k} fold={i}/{numFolds} trial={trial}/{numTrials} --")
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_i, X_train_i, verbose=verb, batch_size=batchSize, epochs=numEpochs)
            
            # EVALUATION: Compare reconstructed image to original image
            X_reconstructed = model.predict(X_val_i, verbose=verb)
            mse = np.mean(np.square(X_reconstructed - X_val_i))
            
            if mse < fold_mse_best:
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

def applyFilter(X_data, mask):
    X_filtered = X_data + mask
    return np.clip(X_filtered, 0, 1)

def main():
    # step 1: dataset preparation
    classes = (6, 8, 9)

    (X_data, y_data), (X_test, y_test) = mnist.load_data()

    data_idx = np.where((y_data == classes[0]) | (y_data == classes[1]) | (y_data == classes[2]))[0]
    test_idx = np.where((y_test == classes[0]) | (y_test == classes[1]) | (y_test == classes[2]))[0]

    X_data = X_data[data_idx] /255.0
    X_test = X_test[test_idx] /255.0

    # Flatten images to (N, 784)
    X_data = X_data.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    y_data = y_data[data_idx]
    y_test = y_test[test_idx]
  
    # step 2: parameters
    numEpochs = 15
    batchSize = 100

    k_vec = [12] #[2, 4, 8, 12, 15, 30]

    for k in k_vec:
        # # ---- kfold parameters ----
        # numFolds = 5
        # numTrials = 5 
        # verb = 0    
        # lperc, hperc = 25, 75   
        # getPerformances(X_data, y_data, numFolds, numTrials, batchSize, numEpochs, verb, lperc, hperc, k)

        model = nn.ML4(k)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_data, X_data, verbose=1, batch_size=batchSize, epochs=numEpochs)
        
        X_reconstructed = model.predict(X_data, verbose=1)

        # -------------------------------------------------------------------------
        # # Create the noise mask
        # filter_mask = np.zeros((28, 28))
        # filter_mask[10:19, 10:19] = 1
        # filter_mask = filter_mask.flatten()

        # # Add random salt-and-pepper noise to the mask
        # idx_zeros = np.random.choice(filter_mask.size, size=50, replace=False)
        # filter_mask[idx_zeros] = 0

        # idx_ones = np.random.choice(filter_mask.size, size=50, replace=False)
        # filter_mask[idx_ones] = 1

        filter_mask = np.ones((28, 28))
        filter_mask = filter_mask.flatten()

        X_test_noisy = applyFilter(deepcopy(X_data), filter_mask)
        # -------------------------------------------------------------------------

        X_reconstructed = model.predict(X_test_noisy, verbose=1)
        mse = np.mean(np.square(X_reconstructed - X_data))
        print(f"mse = {mse}")
        
        # ---- prepare autoencoder plot ----
        nr = 6
        nc = 3

        fig1, ax = plt.subplots(nrows=nr, ncols=nc*2, layout="tight", figsize=(8, 10))
        fig1.suptitle(f"k = {k}", fontsize=16, y=0.98)

        idx = 0
        for i in range(nr):
            for j in range(nc):
                # Original Image
                ax[i, 2*j].set_xticks([])
                ax[i, 2*j].set_yticks([])
                ax[i, 2*j].imshow(np.reshape(applyFilter(X_data[idx, :], filter_mask), (28, 28)), cmap='Greys')

                # Reconstructed Image
                ax[i, 2*j+1].set_xticks([])
                ax[i, 2*j+1].set_yticks([])
                ax[i, 2*j+1].imshow(np.reshape(X_reconstructed[idx, :], (28, 28)), cmap='Greys')
                
                idx += 1

        plt.subplots_adjust(top=0.92)   # Adjust layout to make room for the suptitle

        # ---- prepare pattern plot ----
        total_plots = k + 1  # Neurons + Null case
        
        nc = math.ceil(math.sqrt(total_plots))
        nr = math.ceil(total_plots / nc)

        fig2, ax = plt.subplots(nrows=nr, ncols=nc, layout="tight", figsize=(8, 10))
        fig2.suptitle(f"Patterns (k={k})", fontsize=16, y=0.98)

        ax_flat = ax.flatten()

        for p_idx in range(len(ax_flat)):
            if p_idx < total_plots:
                patternType = np.zeros(k)
                
                if p_idx < k:
                    # Case: Individual Neurons
                    patternType[p_idx] = 1.0
                    title = f"Neuron {p_idx}"
                else:
                    # Case: All zeros
                    title = "Null"

                # Generate and show image
                pattern_img = model.showPattern(patternType)
                ax_flat[p_idx].imshow(np.reshape(pattern_img, (28, 28)), cmap='Greys')
                ax_flat[p_idx].set_title(title, fontsize=10)
                ax_flat[p_idx].axis('off')
            else:
                # Hide any remaining unused subplots in the grid
                ax_flat[p_idx].axis('off')

        plt.subplots_adjust(top=0.90)
        
        # ---- plot ----
        plt.show() # This will open both fig1 and fig2 in separate windows

if __name__ == '__main__':
    main()
    