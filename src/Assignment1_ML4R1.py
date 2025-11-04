from pandas import read_csv
import numpy as np
import math 

def main():
    runs = 100              # numero di test
    split = 0.7             # percentuale di dati che vanno nel training
    smooth_param = 1 # 1e-3     # parametro per gaussian smoothing - - - !!! NON INSERIRE ZERO !!! - - -
    datasetName = 'wdbc'    # opzioni: 'breast_cancer', 'weather', 'iris, 'wdbc'

    # nota: in weather un smooth_param = 1 migliora (di poco) le performace (missing data)

    accuracies = []

    for _ in range(runs):
        data, _ = readData_csv(datasetName) 
        T_train, Y_train, T_test, Y_test = divide_data(data, split)

        nb = nbayes(smooth_param)
        nb.fit(T_train, Y_train)
        acc = nb.test(T_test, Y_test)
        accuracies.append(acc * 100)

    # --- Risultati  ---
    print("Result summary for {" +datasetName +"}\nnumber of runs:\t\t" +str(runs))
    print(f"Average Accuracy:\t{np.mean(accuracies):.2f}")
    print(f"Standard Deviation:\t{np.std(accuracies):.2f}")

def isNaN(x):
    if isinstance(x, str):
        return x.lower() == 'nan'
    if isinstance(x, float):
        return math.isnan(x)
    return False

def readData_csv(dataset):
    # opzioni: 'breast_cancer', 'weather', 'wdbc', 'iris'
    if dataset == 'breast_cancer':
        data = read_csv('breast_cancer.txt', sep=',').to_numpy()
    elif dataset == 'weather':
        data = read_csv('weather_dataset.txt', sep='\\s+').to_numpy()
    elif dataset == 'iris':
        data = read_csv('iris.txt', sep=',').to_numpy()
    elif dataset == 'wdbc':
        data_unformatted = read_csv('wdbc.txt', sep=',').to_numpy()
        data = []
        for i, entry_i in enumerate(data_unformatted[:, 1]):
            joined = np.append(data_unformatted[i, 2:], entry_i)
            data.append(list(joined))
        data = np.array(data)
    else:
        print("Dataset " +str(dataset) + ".txt not in this folder")
    
    # in weather_dataset capita che l'ultima riga sia una nan, in tal caso la elimino
    if isNaN(data[1, -1]): 
        data = data[:, :-1]

    clean_data = []
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

class nbayes:

    def __init__(self, gauss_smooth_par=1e-3):
        self.a = gauss_smooth_par
        self.trained = False
        self.y = []
        self.P_x_y_i = []
        self.P_t = []

    def fit(self, X_train, y_train): # Nota: in caso di dati numerici ipotizzo distribuzione di probabilità normale

        y = list(set(y_train)) # estrae tutti i vaolri unici delle y e li mette in una lista  
        
        ## 1- creo la struttura per P_t e P_x_y_i vuote
        P_t = {a : 0 for a in y} # priorità a priori di y
        noe = len(X_train)

        n_col = X_train.shape[1]
        P_x_y_i = {y_i: [{} for _ in range(n_col)] for y_i in y} # priorità condizionata di X quando y_i è verificato
        
        ## 2 - inizializzo all'interno di P_x_y_i i valori numerici
        for i, entry_i in enumerate(X_train[0, :]):
            '''
            Provo a vedere se la variabile in ingresso può essere convertita in float.
            Se è così predispongo la colonna i-esima per contenere:
                * somma di tutti i valori letti
                * quadrato della somma (mi serve per calcolare media e deviazione standard)
                * numero di valori letti (same)
            '''
            if isinstance(entry_i, bool) == False:
                try:
                    Test = float(entry_i)
                    for y_i in y:
                        P_x_y_i[y_i][i].update({'values': [], 'count': 0})
                except (ValueError, TypeError):
                    pass

        ## 3 - popolo P_x_y_i e P_t
        for i, row in enumerate(X_train):
            y_i = y_train[i]
            P_t[y_i] += 1
            for j, entry_j in enumerate(row):
                if 'values' in P_x_y_i[y_i][j]:  # variabile numerica
                    entry = float(entry_j)     
                    P_x_y_i[y_i][j]['values'].append(entry) 
                    P_x_y_i[y_i][j]['count'] += 1
                else:   # variabile categoriale
                    if entry_j not in P_x_y_i[y_i][j]:
                        for yj in y: # deve aggiungere la variabile anche in tutte le altre classi
                            P_x_y_i[yj][j].update({entry_j : 0})
                    P_x_y_i[y_i][j][entry_j] += 1

        ## 4 - normalizzo P_x_y_i
        for y_i in y:                                   # per ogni risultato
            for column_j in range(len(P_x_y_i[y_i])):   # per ogni colonna di P_X_y_i
                # dati numerici
                if 'values' in P_x_y_i[y_i][column_j]:
                    N = P_x_y_i[y_i][column_j]['count']

                    if N == 0: # caso in cui per quel risultato particolare y_i non ci sono dati numerici disponibili
                        mean = 0
                        stdev = 1e-9
                    else:
                        # calcolo media
                        mean = sum(P_x_y_i[y_i][column_j]['values'])/N
                        
                        # calcolo deviazione standard
                        stdev = 0
                        for x_i in P_x_y_i[y_i][column_j]['values']:
                            stdev += (x_i - mean) ** 2 
                        stdev = math.sqrt(1/(N-1) * stdev)


                    P_x_y_i[y_i][column_j] = {'mean': mean, 'var': stdev}

                # dati categoriali
                else:
                    total_count = sum(P_x_y_i[y_i][column_j].values())
                    nu = len(P_x_y_i[y_i][column_j])  # numero di possibili categorie

                    for category in P_x_y_i[y_i][column_j]:
                        # smoothing
                        P_x_y_i[y_i][column_j][category] = (P_x_y_i[y_i][column_j][category] + self.a) / (total_count + self.a * nu)

        # 5 - calcolo P_t
        for y_i in y:
            P_t[y_i] = P_t[y_i]/noe
        
        self.y = list(set(y_train))  # tipologia di possibili outcome
        self.P_x_y_i = P_x_y_i       # probabilità di X quando y_i è verificato
        self.P_t = P_t               # probabilità a priori di y (in questa tipologia di problemi y===t)
        self.trained = True

    def predict(self, X_test):
        P_x_y_i = self.P_x_y_i    
        P_t = self.P_t          
        y = self.y              

        # densità di probabilità gaussiana
        def dpG(x, mean, var):
            ''' 
            in una distribuzione di probabilità gaussiana, P(x) = 0 per ogni x puntuale, 
            pertanto devo definire un intervallo deta << variazione_standard e calcolare la probabilità che la x cada in quello specifico intervallo
            '''
            delta = 1e-10   # già 1e-6 è buono, se lo riduco ancora prendo qualche decimo di percentuale
            coeff = 1 / math.sqrt(2 * math.pi * (var + delta))
            exponent = math.exp(-((x - mean) ** 2) / (2 * (var + delta)))
            return coeff * exponent

        g = {y_i: math.log(P_t[y_i]) for y_i in y}  # probabilità a priori di y (in logaritmico)

        for y_i in y:
            for j, x_t_i in enumerate(X_test):
                # Numeric feature
                if 'mean' in P_x_y_i[y_i][j]:
                    mean = P_x_y_i[y_i][j]['mean']
                    var  = P_x_y_i[y_i][j]['var']
                    x_t_i = float(x_t_i)    # in alcuni casi mi dava errory perchè risultava "numpy.str_"
                    prob = dpG(x_t_i, mean, var)

                # Categorical feature
                else:
                    prob = P_x_y_i[y_i][j].get(x_t_i, 1e-9)  # nel caso non dovesse esserci x_t_i

                # per le proprietà dei logaritmi, la produttoria diventa sommatoria
                g[y_i] += math.log(prob + 1e-9)

        # Return the class with the maximum log probability
        best_class = max(g, key=g.get)
        return best_class
        
    def test(self, X_test, y_test):
        if not self.trained:
            print("something went wrong") #raise valueError

        y_predict = []
        for x_i in X_test:
            y_predict.append(self.predict(x_i))
        return (np.array(y_test) == np.array(y_predict)).sum()/len(y_test)

if __name__ == '__main__':
    main()