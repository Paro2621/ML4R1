import numpy as np
from statistics import mode

class kNN:
    def __init__(self, k):
        if isinstance(k, int):
            self.k = [k]
        else:
            self.k = k
        self.trained = False 

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.trained = True

    def predict(self, x_test):
        k = self.k
        k_max = max(k)

        predictions = []

        # Initialize with +inf distances
        dist_k = [float('inf')] * k_max
        y_k = [-1] * k_max

        for x_train, y_train in zip(self.X_train, self.y_train):
            diff = x_test - x_train
            dist = np.sqrt(np.sum(diff * diff))

            # Find worst current neighbor
            max_idx = dist_k.index(max(dist_k))

            if dist < dist_k[max_idx]:
                dist_k[max_idx] = dist
                y_k[max_idx] = y_train

        pairs = sorted(zip(dist_k, y_k))
        y_k = [y for _, y in pairs]

        for k_i in self.k:
            predictions.append(mode(y_k[0:k_i]))

        return predictions

    def test(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Classifier not trained")

        k_count = len(self.k)
        correct = np.zeros(k_count, dtype=int)

        for x_i, y_true in zip(X_test, y_test):
            preds = self.predict(x_i)

            for j, p in enumerate(preds):
                if p == y_true:
                    correct[j] += 1

        return correct / len(y_test)