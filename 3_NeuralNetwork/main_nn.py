from pandas import read_csv
import keras as k
import math
import numpy as np

class ML3(k.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = k.layers.Dense(12, activation="sigmoid", input_shape=(16,))
        self.dense2 = k.layers.Dense(2, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def get_config(self):
        config = super().get_config()
        # [!] se passi argomenti custom nell'initializer ricordati di inserirli qui !!!
        return config

def main():
    # step 1: dataset preparation
    data, _ = readData_csv('data')
    data = data[np.random.permutation(data.shape[0]), :]
    
    # step 2: parameters for training algorithm
    numFolds = 5
    numTrials = 5
    numEpochs = 15

    batchSize = 100

    # optional parameters
    verb = 1
        # 0 = no output, 1 = progress bar, 2 = one-line per epoch

    # set up evaluation parameters
    msevals = []            # store a list of objective/metric values
    lperc, hperc = 25, 75   # (range of percentiles to be used in final reporting)
        
    # initialize the model
    model = ML3()

    msebest = None
    best_model = None

    for i in range(numFolds):
        print(f"FOLD {i+1}/{numFolds}")
        
        fr = len(data)/numFolds # fold ratio 
        idx_0 = math.ceil(i*fr)
        idx_f = math.floor((i+1)*fr)

        idx_test = range(idx_0,idx_f)
        idx_train = list(range(idx_0))+list(range(idx_f,len(data)))

        data_train = normalize_data(data[idx_train, :])
        data_test = normalize_data(data[idx_test, :])

        X_train_i = data_train[idx_train, :-2]
        Y_train_i = data_train[idx_train, -2:]

        X_test_i = data_test[idx_test, :-2]
        Y_test_i = data_test[idx_test, -2:]

        # MULTI-START TRAINING LOOP
        # perform training "numTrials" times on the same training/test split,
        # starting from different initialisations, and keep the best loss/metric value
        for trial in range(numTrials):
            model.name = f"ML3_{trial+1}" 
                # trial parte da zero ma io voglio che il primo modello si chiami ML3_1

            # TRAINING
            print(f"training start {trial+1}/{numTrials}")
            # to start training it is necessary to "compile" first
            model.compile(optimizer = 'adam', loss = k.losses.MeanSquaredError())
            model.fit(X_train_i, Y_train_i, verbose = verb, batch_size = batchSize, validation_split = .1, epochs = numEpochs)
            print(f"training complete {trial+1}/{numTrials}")

            # CROSS-VALIDATION
            print("--validation--")
            # estimate performance index (loss or other metric) on the test set
            Y_pred = model.predict(X_test_i, verbose = verb)

            mse = ((Y_pred - Y_test_i)**2).sum()/(len(Y_pred)*2)
            
            if msebest is None or msebest > mse:
                msebest = mse
                model.save('tmp/best.keras')
            
            # reset model to new initial state for the next iteration
            model = k.models.clone_model(model)

        # add the best test value to the list
        msevals.append(msebest)

    best_model = k.models.load_model('tmp/best.keras', custom_objects={'ML3': ML3})
    msevals = np.array(msevals, dtype=float)
    print(msevals)
    print("")

    # use collected values to compute median and percentiles
    # the median will be the estimated performance
    # the interval [25th, 75th] will be the range where the performance
    # occurs half of the time
    if len(msevals)>0:
        low, med, high = np.percentile(msevals, (lperc, 50, hperc))
        print(f"mse = {med:.3E} (typical)\nmse in [{low:.3E}, {high:.3E}] with probability >= {(hperc-lperc)/100.:.2f}")
        print(f"\nbest model\n\tname = {best_model.name}\n\tmse = {msebest:.3E}")

def normalize_data(data):
    nr, nc = data.shape
    for i in range(nc):
        col = data[:, i]
        minval = col.min()
        maxval = col.max()

        if maxval == minval:
            data[:, i] = 0.0
        else:
            data[:, i] = (col - minval) / (maxval - minval)

    return data
            
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
[0.08853957 0.00082806 0.01135375 0.00203993 0.27105985]
mse = 1.135E-02 (typical)
mse in [8.854E-02, 2.040E-03] with probability >= 0.50

[0.08786808 0.08750053 0.08186375 0.08186375 0.08186375]
mse = 8.186E-02 (typical)
mse in [8.186E-02, 8.750E-02] with probability >= 0.50
'''