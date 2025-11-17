import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

nr = 5
nc = 5
N = X_train.shape[0]
fig1, ax = plt.subplots(nrows=nr, ncols=nc,layout="tight", figsize=(7,8))
fig1.tight_layout()
for i in range(nr):
    for j in range(nc):
        ax[i,j].set_yticks([])
        ax[i,j].set_xticks([])
        l = np.random.randint(N)
        ax[i,j].imshow(np.reshape(X_train[l,:],(28,28)),cmap='Greys')
plt.show()