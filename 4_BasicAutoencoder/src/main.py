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

def mnist_load_normalize(classes):
    (X_data, y_data), (X_test, y_test) = mnist.load_data()

    data_idx = np.where(np.isin(y_data, classes))[0]
    test_idx = np.where(np.isin(y_test, classes))[0]

    X_data = X_data[data_idx] / 255.0
    X_test = X_test[test_idx] / 255.0

    X_data = X_data.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    y_data = y_data[data_idx]
    y_test = y_test[test_idx]

    return (X_data, y_data), (X_test, y_test)

def kFold_mnist_autoencoder(X_data, numFolds, currentFold):
    dpf = X_data.shape[0]/numFolds # data per fold

    idx_0 = int(currentFold*dpf)
    idx_f = int((currentFold+1)*dpf)

    idx_validation = range(idx_0,idx_f)
    idx_train = list(range(idx_0)) + list(range(idx_f,X_data.shape[0]))

    X_train = X_data[idx_train].astype(float)

    X_validation = X_data[idx_validation].astype(float)

    return [X_train, X_validation]

def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.set_weights([
                layer.kernel_initializer(layer.kernel.shape),
                layer.bias_initializer(layer.bias.shape)
            ])

def getPerformances(model, X_data, numFolds, numTrials, batchSize, numEpochs, verb, lperc, hperc, k):
    msevals = []  
    for i in range(numFolds):
        fold_mse_best = float('inf') 

        [X_train_i, X_val_i] = kFold_mnist_autoencoder(deepcopy(X_data), numFolds, i)

        for trial in range(numTrials):
            tf.keras.backend.clear_session()
            
            print(f"-- k={k} fold={i}/{numFolds} trial={trial}/{numTrials} --")

            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_i, X_train_i, verbose=verb, batch_size=batchSize, epochs=numEpochs)
            
            X_reconstructed = model.predict(X_val_i, verbose=verb)
            mse = np.mean(np.square(X_reconstructed - X_val_i))
            
            if mse < fold_mse_best:
                fold_mse_best = mse
            
            # Reset model
            reset_weights(model)

        msevals.append(fold_mse_best)

    msevals = np.array(msevals, dtype=float)
    
    # Print results
    if len(msevals)>0:
        print(f"--- Result Summary for k = {k} ---")
        print(msevals) 
        low, med, high = np.percentile(msevals, (lperc, 50, hperc))
        print(f"mse = {med:.3F} (typical)\naccuracy in [{low:.3F}, {high:.3F}] with probability >= {(hperc-lperc)/100:.2F}")

def applyFilter(X_data, mask):
    X_filtered = X_data + mask
    return np.clip(X_filtered, 0, 1)

def autoencoderPlot(k, X_test_noisy, X_reconstructed):
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
            ax[i, 2*j].imshow(np.reshape(X_test_noisy[idx, :], (28, 28)), cmap='Greys')

            # Reconstructed Image
            ax[i, 2*j+1].set_xticks([])
            ax[i, 2*j+1].set_yticks([])
            ax[i, 2*j+1].imshow(np.reshape(X_reconstructed[idx, :], (28, 28)), cmap='Greys')
            
            idx += 1

    plt.subplots_adjust(top=0.92)

def patternPlot(k, model):
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
                patternType[p_idx] = 1.0
                title = f"Neuron {p_idx}"
            else:
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

def main():
    # step 1: dataset preparation
    classes = (3, 8) #(6, 8, 9)

    (X_data, _), (X_test, _) = mnist_load_normalize(classes)
  
    # step 2: parameters
    verb = 2  
    numEpochs = 15
    batchSize = 100

    k_vec = [12]

    for k in k_vec:
        model = nn.ML4(k)

        # ---- kfold parameters ----
        numFolds = 5
        numTrials = 5 
        lperc, hperc = 25, 75   
        getPerformances(model, X_data, numFolds, numTrials, batchSize, numEpochs, verb, lperc, hperc, k)
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_data, X_data, verbose=0, batch_size=batchSize, epochs=numEpochs)

        # -------------------------------------------------------------------------
        # Create the noise mask
        filter_mask = np.zeros((28, 28))
        filter_mask[12:21, 12:21] = 1
        filter_mask = filter_mask.flatten()

        # Add salt-and-pepper noise
        idx_zeros = np.random.choice(filter_mask.size, size=50, replace=False)
        filter_mask[idx_zeros] = 0

        idx_ones = np.random.choice(filter_mask.size, size=50, replace=False)
        filter_mask[idx_ones] = 1

        X_test_noisy = applyFilter(deepcopy(X_test), filter_mask)
        # -------------------------------------------------------------------------
        
        print("\nmse for plain data")
        X_reconstructed = model.predict(deepcopy(X_test), verbose=1)
        mse = np.mean(np.square(X_reconstructed - X_test))
        print(f"mse = {mse}")

        print("\ntest w/ noisy data")
        X_reconstructed = model.predict(deepcopy(X_test_noisy), verbose=1)
        mse = np.mean(np.square(X_reconstructed - X_test))
        print(f"mse  = {mse}")
        
        # ---- plot ----
        autoencoderPlot(k, X_test_noisy, X_reconstructed)
        patternPlot(k, model)
        plt.show() # This will open both fig1 and fig2 in separate windows

if __name__ == '__main__':
    main()
    