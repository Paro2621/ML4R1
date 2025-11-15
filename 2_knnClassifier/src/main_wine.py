from pandas import read_csv
from statistics import mode
import math
import numpy as np
import classifier

def main():
    runs = 100               # numero di test
    split = 0.7             # percentuale di dati che vanno nel training
    
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
            # data = preprocess_nKK(data, format)
            T_train, Y_train, T_test, Y_test = divide_data(data, split)

            kNNc = classifier.kNN(k)
            kNNc.fit(T_train, Y_train)
            acc = kNNc.test(T_test, Y_test)
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
            zero_one = np.zeros((len(column), len(uniques)))

            for k, val in enumerate(uniques):
                zero_one[:, k] = (column == val)

            formattedData.append(zero_one)

        elif ftype == 'co': # ategorical ordinal -> label encoding
            # TODO: implement function
            pass

    formattedData.append(data[:, -1])
    return np.column_stack(formattedData)

if __name__ == '__main__':
    main()

'''
    BEFORE NORMALIZATION
        Result summary for {wine}
        number of runs: 100
        k       accuracy        std_dev
        ———————————————————————————————
        1       73.741          4.985
        2       73.222          5.835
        3       71.704          5.581
        4       70.889          5.361
        5       70.704          5.477
        10      69.130          5.251
        15      69.093          5.541
        20      69.852          4.765
        30      72.093          5.279
        40      69.759          5.146
        50      70.500          5.160

    AFTER NORMALIZATION
        Result summary for {wine}
        number of runs: 100
        k       accuracy        std_dev
        ———————————————————————————————
        1       95.019          2.396
        2       94.833          2.596
        3       95.778          2.444
        4       95.981          2.633
        5       95.833          2.333
        10      96.556          2.095
        15      97.074          2.271
        20      96.648          2.171
        30      96.704          1.935
        40      96.074          2.700
        50      95.111          3.104
'''