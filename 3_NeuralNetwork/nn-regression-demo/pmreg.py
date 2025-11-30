from pandas import read_csv
import numpy as np

# read the data
data = read_csv('nn-regression-demo/UCI-CBM/data.txt', sep='\s+').values
c = 2
d = data.shape[1]-c
n = data.shape[0]

# permute your data (must be i.i.d. anyway)
data = data[np.random.permutation(n), :]

# set up algorithm parameters
kfolds =  5
num_trials = 5
num_epochs = 15
split_frac = 1./kfolds

# set up model hyperparameters (shallow model --> only the number h of hidden units)
h = 12

# set up evaluation parameters
# (range of percentiles to be used in final reporting)
lperc, hperc = 25, 75

# prepare the model
import tensorflow.keras as tf_keras
model=tf_keras.Sequential(name='my_network')
model.add(tf_keras.layers.Dense(h,name='Layer_1',activation='sigmoid',input_shape=(d,)))
model.add(tf_keras.layers.Dense(c,name='Layer_2',activation='linear')) # if "linear", the activation can be omitted

# store a list of objective/metric values
msevals = []

# K-FOLD CROSS-VALIDATION LOOP
# perform "kfolds" different trainings each with an independent test subset
for k in range(kfolds):
  try:
    print(f"=== {k+1} of {kfolds} ===")
    
    # splitting data by selecting appropriate indexes
    # test set is the subset from idxmin (including) to idxtop (excluding)
    # compute indexes first, then use them to select test subset
    idxmin = int(1.*n*k/kfolds)
    idxtop = int(1.*n*(k+1)/kfolds)
    n_test = idxtop-idxmin
    print(f"{idxmin} {idxtop}")
    X_test = data[idxmin:idxtop,:d]
    Y_test = data[idxmin:idxtop,-c:]
    # training set is the data before and after the test subset
    # again compute indexes first, then use them to select training subset
    idxtrain = list(range(idxmin))+list(range(idxtop,n))
    X_train = data[idxtrain,:d]
    Y_train = data[idxtrain,-c:]

    # normalise your data
    # NOTE we are also normalising targets to make it easier to train for regression and
    #      to interpret the loss value
    minXvals = X_train.min(axis=0)
    Xranges = X_train.max(axis=0)-minXvals
    for i in range(d):
      if Xranges[i] != 0:
        X_train[:,i] = (X_train[:,i] - minXvals[i]) / Xranges[i]
        X_test[:,i] = (X_test[:,i] - minXvals[i]) / Xranges[i]
    minYvals = Y_train.min(axis=0)
    Yranges = Y_train.max(axis=0)-minYvals
    for i in range(c):
      if Yranges[i] != 0:
        Y_train[:,i] = (Y_train[:,i] - minYvals[i]) / Yranges[i]
        Y_test[:,i] = (Y_test[:,i] - minYvals[i]) / Yranges[i]

    # MULTI-START TRAINING LOOP
    # perform training "num_trials" times on the same training/test split,
    # starting from different initialisations, and keep the best loss/metric value
    msebest = None
    for trial in range(num_trials):
      print(f"--- trial no. {trial+1}/{num_trials} ---")

      # TRAINING
      # to start training it is necessary to "compile" first
      model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())
      model.fit(X_train, Y_train, batch_size=100, validation_split=.1, epochs=num_epochs)

      # CROSS-VALIDATION
      # estimate performance index (loss or other metric) on the test set
      Y_pred = model.predict(X_test)
      mse = ((Y_pred - Y_test)**2).sum()/(n_test*c)
      if msebest is None or msebest > mse:
        msebest = mse
      # reset model to new initial state for the next iteration
      model = tf_keras.models.clone_model(model)

    # add the best test value to the list
    msevals.append(msebest)

  # trick to avoid error messages if you press control-c to stop mid-training
  # (remember "try:" at beginning of loop body)
  except:
    break
    
print(msevals)
print("")

# use collected values to compute median and percentiles
# the median will be the estimated performance
# the interval [25th, 75th] will be the range where the performance
# occurs half of the time
if len(msevals)>0:
  m, r, p = np.percentile(msevals,(hperc,50,lperc))
  print(f"mse = {r:.3E} (typical), mse in [{m:.3E}, {p:.3E}] with probability >= {(hperc-lperc)/100.:.2f}")
