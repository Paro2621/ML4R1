'''
Create a python virtual environment:
1. create a folder and open a terminal 
2. run "python3 -m venv ." (the . in this case is the path)
3. enter in that folder and run "source bin/activate"
3. from here i can then run "./pip install tensorflow"

MEMO: how to run the script in a python virtual environment
1. open vsCode
2. click on the searchbar
3. ">Python: select interpreter"
4. --> enter interpreter path
5. inserire il percorso del virtual environment
'''

import pandas as pd
import numpy as np
import math
from statistics import mode
from tensorflow.keras.datasets import mnist
import classifier

def main():
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
    main()