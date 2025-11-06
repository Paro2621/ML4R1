from pandas import read_csv
import numpy as np
import math
from statistics import mode

def main():
    runs = 10               # numero di test
    split = 0.9             # percentuale di dati che vanno nel training
    
    datasetName = 'wine'
        # [!] Attention: for wine, the first entry is y_i: 
        # [!] DO NOT CONSIDER THAT COLUMN WHEN WRITING THE FORMAT  

    format = ['nn' for _ in range(13)]
        # 'ns' -> numerical             -> standard distribution
        # 'nn' -> numerical             -> normalization
        # 'co' -> categorical ordinal   -> label encoding (DATASET MUST BE PREPROCESSED)
        # 'cn' -> categorical nominal   -> 0-1-encoding

    k_vector = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

    accuracies = []
    stds = []

    accuracy_i = []

    for k in k_vector:
        accuracy_i = []  # reset for each k
        for i in range(runs):
            data, _ = readData_csv(datasetName) 
            data = preprocess_nKK(data, format)
            T_train, Y_train, T_test, Y_test = divide_data(data, split)

            nKK = nKKclassifier(k)
            nKK.fit(T_train, Y_train)
            acc = nKK.test(T_test, Y_test)
            accuracy_i.append(acc * 100)

        accuracies.append(np.mean(accuracy_i))
        stds.append(np.std(accuracy_i))

    # --- Risultati  ---
    print("Result summary for {" +datasetName +"}\nnumber of runs:\t" +str(runs))
    print("k\taccuracy\tstd_dev")
    print("———————————————————————————————")
    for k, acc, std in zip(k_vector, accuracies, stds):
        print(f"{k}\t{acc:.3f}\t\t{std:.3f}")

def isNaN(x):
    if isinstance(x, str):
        return x.lower() == 'nan'
    if isinstance(x, float):
        return math.isnan(x)
    return False

def readData_csv(dataset):
    if dataset == 'wine': # puts the output in the last column
        data_unformatted = read_csv('wine.txt', sep=',').to_numpy().astype(float)
        data = []
        
        for i, entry_i in enumerate(data_unformatted[:, 0]):
            joined = np.append(data_unformatted[i, 1:], entry_i)
            data.append(list(joined))

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

def divide_data(data, split):
    np.random.shuffle(data)

    noe = len(data)
    splitIndex = int(noe*split)

    T_train = data[0:splitIndex, 0:-1]
    Y_train = data[0:splitIndex, -1]

    T_test = data[splitIndex:, 0:-1]
    Y_test = data[splitIndex:, -1]

    return [T_train, Y_train, T_test, Y_test]

def preprocess_nKK(data, format):
    formattedData = []

    for i in range(data.shape[1]-1):
        column = data[:, i].astype(float)
        ftype = format[i]

        if ftype == 'ns': # numerical -> standardization
            mean = column.mean()
            std = column.std()
            formattedData_i = (column - mean) / std
            formattedData.append(formattedData_i)

        elif ftype == 'nn': # numerical -> normalization
            minv = column.min()
            maxv = column.max()
            formattedData_i = (column - minv) / (maxv - minv)
            formattedData.append(formattedData_i)

        elif ftype == 'cn': # categorical nominal -> 0-1-encoding
            uniques = np.unique(column)
            onehot = np.zeros((len(column), len(uniques)))

            for k, val in enumerate(uniques):
                onehot[:, k] = (column == val)

            formattedData.append(onehot)

        elif ftype == 'co': # ategorical ordinal -> label encoding
            # TODO: implement function
            pass

    formattedData.append(data[:, -1])

    return np.column_stack(formattedData)
    

class nKKclassifier:

    def __init__(self, k):
        self.k = k
        self.trained = False 

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.trained = True
        
    def predict(self, x_test):
        k = self.k
        y_k = [-1 for _ in range(k)]
        dist_k = [float('inf') for _ in range(k)]

        # Compute distance from x_test to each training sample
        for x_train, y_train in zip(self.X_train, self.y_train):
            dist = 0.0

            for a, b in zip(x_test, x_train):
                diff = float(a) - float(b)
                dist += diff * diff
            dist = math.sqrt(dist)

            # Keep the k closest
            max_idx = dist_k.index(max(dist_k))
            if dist < dist_k[max_idx]:
                dist_k[max_idx] = dist
                y_k[max_idx] = y_train

        return mode(y_k)

    def test(self, X_test, y_test):
        if not self.trained:
            print("something went wrong") #raise valueError

        y_predict = []
        for x_i in X_test:
            y_predict.append(self.predict(x_i))

        return (np.array(y_test) == np.array(y_predict)).sum()/len(y_test)

if __name__ == '__main__':
    main()

    '''
    BEFORE NORMALIZATION
        **Result summary for `{wine}`**  
        Number of runs: 10  

        | k  | Accuracy (%) | Std. Dev. |
        |:--:|:-------------:|:---------:|
        | 1  | 78.889        | 10.482    |
        | 2  | 72.778        | 7.638     |
        | 3  | 72.778        | 6.781     |
        | 4  | 64.444        | 9.027     |
        | 5  | 63.889        | 8.333     |
        | 10 | 71.667        | 11.235    |
        | 15 | 74.444        | 7.115     |
        | 20 | 73.333        | 9.876     |
        | 30 | 76.111        | 9.313     |
        | 40 | 71.111        | 7.778     |
        | 50 | 72.222        | 11.111    |

    AFTER NORMALIZATION
        **Result summary for `{wine}`**  
        Number of runs: 10  

        | k  | Accuracy (%) | Std. Dev. |
        |:--:|:-------------:|:---------:|
        | 1  | 95.556        | 4.843     |
        | 2  | 93.333        | 6.479     |
        | 3  | 96.111        | 2.546     |
        | 4  | 97.778        | 3.685     |
        | 5  | 93.889        | 5.800     |
        | 10 | 97.778        | 3.685     |
        | 15 | 97.222        | 3.727     |
        | 20 | 96.667        | 5.092     |
        | 30 | 97.778        | 2.722     |
        | 40 | 96.667        | 4.444     |
        | 50 | 93.889        | 5.241     |
    '''