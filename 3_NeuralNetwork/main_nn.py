from pandas import read_csv
import keras as k
import math
import numpy as np

class ML3(k.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = k.layers.Dense(12, activation="sigmoid", input_shape=(16,))
        # self.dense1 = k.layers.Dense(8, activation="relu")
        self.last = k.layers.Dense(2, activation="linear")

    def call(self, inputs):
        x = self.first(inputs)
        # x = self.dense1(x)
        # ricordati di connettere i layer interni
        return self.last(x)
    
    def get_config(self):
        config = super().get_config()
        # [!] se passi argomenti custom nell'initializer ricordati di inserirli qui !!!
        return config

def split_normalize(data, numFolds, currentFold):
    dpf = data.shape[0]/numFolds # data per fold

    idx_0 = int(currentFold*dpf)
    idx_f = int((currentFold+1)*dpf)

    print(f"{idx_0} -> {idx_f}")

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

def main():
    # step 1: dataset preparation
    data, _ = readData_csv('data')
    data = data[np.random.permutation(data.shape[0]), :]
    
    # step 2: parameters
    numFolds = 5
    numTrials = 5
    numEpochs = 15
    batchSize = 100
    verb = 0

    msevals = []            
    lperc, hperc = 25, 75   

    for i in range(numFolds):
        print(f"FOLD {i+1}/{numFolds}")

        fold_mse_best = None 

        [X_train_i, Y_train_i, X_test_i, Y_test_i] = split_normalize(data.copy(), numFolds, i)
     
        for trial in range(numTrials):
            # initialize the model
            model = ML3()

            model.name = f"ML3_F{i+1}T{trial+1}" 

            print(f"training start {trial+1}/{numTrials}")
            model.compile(optimizer = 'adam', loss = k.losses.MeanSquaredError())
            model.fit(X_train_i, Y_train_i, verbose = verb, batch_size = batchSize, validation_split = .1, epochs = numEpochs)
            
            # EVALUATION
            Y_pred = model.predict(X_test_i, verbose = verb)
            mse = ((Y_pred - Y_test_i)**2).sum()/(len(Y_pred)*2)
            
            # Update fold best
            if fold_mse_best is None or mse < fold_mse_best:
                fold_mse_best = mse
            
            # Reset model
            del model

        msevals.append(fold_mse_best)

    msevals = np.array(msevals, dtype=float)
    
    # Print results
    print(msevals) 
    print("")

    if len(msevals)>0:
        low, med, high = np.percentile(msevals, (lperc, 50, hperc))
        print(f"mse = {med:.3E} (typical)\nmse in [{low:.3E}, {high:.3E}] with probability >= {(hperc-lperc)/100.:.2f}")
         
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

if __name__ == '__main__':
    main()

'''
[0.08700113 0.00237802 0.00067161 0.00037403 0.49639787]
mse = 2.378E-03 (typical)
mse in [8.700E-02, 6.716E-04] with probability >= 0.50

[0.08473287 0.08360496 0.0876412  0.08818239 0.08595251]
mse = 8.595E-02 (typical)
mse in [8.473E-02, 8.764E-02] with probability >= 0.50

best global model
        mse = 8.360E-02
        name: ML3_F2T5
'''