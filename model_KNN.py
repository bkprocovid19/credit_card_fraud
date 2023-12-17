import numpy as np
from collections import Counter

#Dữ liệu đầu vào là ma trận trong numpy
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

class KNN:
    def __init__(self, k=3, metric='euclidean_distance'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        if (self.metric == 'euclidean_distance'):
            distances = np.linalg.norm(self.X_train - x, axis=1)
        if (self.metric == 'manhattan_distance'):
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        if (self.metric == 'chebyshev_distance'):
            distances = [chebyshev_distance(x, x_train) for x_train in self.X_train]
        
        # k láng giềng gần nhất
        k_nearest_distances = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_distances] 

        # Số nhãn xuất hiện nhiều nhất trong k láng giềng
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions