"""
TODO: scrivi mail al prof per riga 82 di main_nn -> assignment 3
"""

import numpy as np
import keras as ke
import matplotlib.pyplot as plt
import math

from tensorflow.keras.datasets import mnist
from copy import deepcopy

class ML4(ke.Model):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        # ENCODER: Compress data
        self.bottleneck = ke.layers.Dense(h, activation="relu")
        # DECODER: Reconstruct data
        self.output_layer = ke.layers.Dense(784, activation="sigmoid")

    def call(self, inputs):
        x = self.bottleneck(inputs)
        return self.output_layer(x)
    
    def showPattern(self, input):
        pattern = np.array(input).reshape(1, -1)
        return self.output_layer(pattern)
    
    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h})
        return config

class ML4_deep(ke.Model):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        # ENCODER: Compress data
        self.encoder_hidden = ke.layers.Dense(128, activation="relu")
        self.bottleneck = ke.layers.Dense(h, activation="relu")
        # DECODER: Reconstruct data
        self.decoder_hidden = ke.layers.Dense(128, activation="relu")
        self.output_layer = ke.layers.Dense(784, activation="sigmoid")

    def call(self, inputs):
        x = self.encoder_hidden(inputs)
        x = self.bottleneck(x)
        x = self.decoder_hidden(x)
        return self.output_layer(x)
    
    def showPattern(self, input):
        pattern = np.array(input).reshape(1, -1)
        x = self.decoder_hidden(pattern)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h})
        return config

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

def testPlot(classes):
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

def main():
    # step 1: dataset preparation
    classes = (4, 2)

    (X_data, y_data), (X_test, y_test) = mnist.load_data()

    data_idx = np.where((y_data == classes[0]) | (y_data == classes[1]))[0]
    test_idx = np.where((y_test == classes[0]) | (y_test == classes[1]))[0]

    X_data = X_data[data_idx] /255.0
    X_test = X_test[test_idx] /255.0

    # Flatten images to (N, 784)
    X_data = X_data.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    y_data = y_data[data_idx]
    y_test = y_test[test_idx]
  
    # step 2: parameters
    numFolds = 5        # 5
    numTrials = 5       # 5
    numEpochs = 15      # 15
    batchSize = 100
    verb = 0
          
    lperc, hperc = 25, 75   

    k_vec = [4, 8, 10, 12, 15]

    for k in k_vec:

        # msevals = []  
        # for i in range(numFolds):
        #     fold_mse_best = float('inf') 

        #     # Ensure kFold_mnist returns X_val_i as images for reconstruction tasks
        #     [X_train_i, y_train_i, X_val_i, y_val_i] = kFold_mnist(deepcopy(X_data), deepcopy(y_data), numFolds, i)

        #     for trial in range(numTrials):
        #         tf.keras.backend.clear_session() # Clears the graph for the new trial
               
        #         model = ML4(k)
        #         print(f"-- k={k} fold={i}/{numFolds} trial={trial}/{numTrials} --")
                
        #         model.compile(optimizer='adam', loss='mse')
        #         model.fit(X_train_i, X_train_i, verbose=verb, batch_size=batchSize, epochs=numEpochs)
                
        #         # EVALUATION: Compare reconstructed image to original image
        #         X_reconstructed = model.predict(X_val_i, verbose=verb)
        #         mse = np.mean(np.square(X_reconstructed - X_val_i))
                
        #         if mse < fold_mse_best:
        #             fold_mse_best = mse
                
        #         # Reset model
        #         del model

        #     msevals.append(fold_mse_best)

        # msevals = np.array(msevals, dtype=float)
        
        # # Print results
        # if len(msevals)>0:
        #     print(f"--- Result Summary for k = {k} ---")
        #     print(msevals) 
        #     low, med, high = np.percentile(msevals, (lperc, 50, hperc))
        #     print(f"mse = {med:.3F} (typical)\naccuracy in [{low:.3F}, {high:.3F}] with probability >= {(hperc-lperc)/100:.2F}\n")

        # Plot
        model = ML4(k)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_data, X_data, verbose=1, batch_size=batchSize, epochs=numEpochs)
        
        X_reconstructed = model.predict(X_data, verbose=1)

        # ---- AUTOENCODER PLOT ----
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
                ax[i, 2*j].imshow(np.reshape(X_data[idx, :], (28, 28)), cmap='Greys')
                if i == 0: ax[i, 2*j].set_title("Orig.")

                # Reconstructed Image
                ax[i, 2*j+1].set_xticks([])
                ax[i, 2*j+1].set_yticks([])
                ax[i, 2*j+1].imshow(np.reshape(X_reconstructed[idx, :], (28, 28)), cmap='Greys')
                if i == 0: ax[i, 2*j+1].set_title("Rec.")

                idx += 1
        
        # Adjust layout to make room for the suptitle
        plt.subplots_adjust(top=0.92) 

# ---- PATTERN PLOT ----
        total_plots = k + 1  # Neurons + Null case
        
        # Calculate grid size to be as square as possible
        nc = math.ceil(math.sqrt(total_plots))
        nr = math.ceil(total_plots / nc)

        fig2, ax = plt.subplots(nrows=nr, ncols=nc, layout="tight", figsize=(8, 10))
        fig2.suptitle(f"Patterns (k={k})", fontsize=16, y=0.96)

        ax_flat = ax.flatten()

        for p_idx in range(len(ax_flat)):
            if p_idx < total_plots:
                patternType = np.zeros(k)
                
                if p_idx < k:
                    # Case: Individual Neurons
                    patternType[p_idx] = 1.0
                    title = f"Neuron {p_idx}"
                else:
                    # Case: All zeros (Null case)
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
        
        # This will open both fig1 and fig2 in separate windows
        plt.show()

if __name__ == '__main__':
    main()
    