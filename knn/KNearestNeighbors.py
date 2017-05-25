import pandas as pd
import numpy as np
from itertools import izip
from sklearn.datasets import make_classification

def euclidean_distance(a, b):
    a,b = np.array(a), np.array(b)
    sq_dist = np.sum((a-b)**2)
    return np.sqrt(sq_dist)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

class KNearestNeighbors(object):
    def __init__(self,k,distance='euclidean_distance',var_type='categorical'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None
        self.var_type = var_type

    def predict(self, X):
        '''
        X = 2d numpy array of features
        y = 1d numpy array of labels

        for each row in X:
            find the index of the K things with the lowest euclidean distance
            find the avg Y val of that thing
            if categorical, round that average to 0 or 1
            else continuous, return the average
        '''
        predictions = []
        for i, row_i in enumerate(X):
            dists = []
            for j, row_j in enumerate(X):
                if i != j:
                    dists.append((euclidean_distance(row_i,row_j),self.y_train[i]))
            dists.sort(key=lambda x: x[0])

            k_neighbor_vals = []
            for kth in range(0,self.k):
                k_neighbor_vals.append(dists[kth][1])

            if self.var_type == 'categorical':
                predictions.append(np.round(np.mean(k_neighbor_vals)))
            else:
                predictions.append(np.mean(k_neighbor_vals))
        return predictions

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

if __name__ == '__main__':
    X, y = make_classification(n_features=4, n_redundant=0, n_informative=1, n_clusters_per_class=1, class_sep=5,random_state=5)
    model = KNearestNeighbors(3)
    model.fit(X,y)
    print "accuracy: ",np.mean(y == model.predict(X))
    print "\tactual\tpredict\tcorrect?"
    for i, (actual, predicted) in enumerate(izip(y, model.predict(X))):
        print "%d\t%d\t%d\t%d" % (i, actual, predicted, int(actual == predicted))
