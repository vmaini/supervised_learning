import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode

class DecisionTree(object):

    def __init__(self, impurity_criterion='entropy'):
        self.root = None  # root Node
        self.feature_names = None

        self.categorical = None  # Boolean array - categorical or continuous var
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array. columns = features, rows = data points
            - y: 1d numpy array of labels
            - feature_names: numpy array of strings
        OUTPUT: None
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        Recursively build the decision tree. Return the root node.
        '''

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        Return the entropy of the array y.
        '''
        classes = np.unique(y)
        total = 0
        for c in classes:
            p_c = float((y == c).sum()) / len(y)
            total += p_c * np.log2(p_c)

        return -1 * total

    def _gini(self, y):
        '''
        Return the gini impurity of the array y.
        '''
        classes = np.unique(y)
        total = 0
        for c in classes:
            p_c = float((y == c).sum()) / len(y)
            total += p_c ** 2
        return 1 - total

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''
        # split_index = 1
        # split_value = 'bat'
        # X = np.array([[1, 'bat'], [2, 'cat'], [2, 'rat'], [3, 'bat'], [3, 'bat']])
        # y = np.array([1, 0, 1, 0, 1])
        full_mat = np.column_stack((X,y))
        X1 = full_mat[full_mat[:,split_index] == split_value][:,0:-1]
        y1 = full_mat[full_mat[:,split_index] == split_value][:,-1]
        X2 = full_mat[full_mat[:,split_index] != split_value][:,0:-1]
        y2 = full_mat[full_mat[:,split_index] != split_value][:,-1]
        return X1, y1.astype(np.int), X2, y2.astype(np.int)

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        y1_weight = len(y1) / float(len(y))
        y2_weight = len(y2) / float(len(y))

        return self._entropy(y) - (y1_weight * self._entropy(y1) + y2_weight * self._entropy(y2))

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits

        our method returns:
        index, value, X1, y1, X2, y2
        '''

        # X = np.array([[1, 'bat'], [2, 'cat'], [2, 'rat'], [3, 'bat'], [3, 'bat']])
        # y = np.array([1, 0, 1, 0, 1])

        gains = []

        for i in range(0, len(X)):
            for j in range(0,len(X[0])):
                X1, y1, X2, y2 = self._make_split(X,y,j,X[i][j])
                gains.append((self._information_gain(y,y1,y2),[i,j]))

        best = max(gains,key=lambda item:item[0])
        best_index = best[1][1]
        best_val = X[best[1][0]][best[1][1]]

        return best_index, best_val, self._make_split(X,y,best_index,best_val)

    def predict(self, X):
        '''
        Return an array of predictions for the feature matrix X.
        '''

        return np.array([self.root.predict_one(row) for row in X])

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)
