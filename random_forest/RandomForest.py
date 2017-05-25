from DecisionTree import DecisionTree
import numpy as np

class RandomForest(object):

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None
        self.root = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_feats):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        trees = [None] * num_trees
        # data = np.concatenate((X,y.reshape(len(X),-1),axis=1))
        for i in range(0,num_trees):
            num_rows = np.shape(X)[0]
            # pick N random indices with replacement, N = len(X)
            sample_indices = np.random.choice(num_rows,num_rows)
            # filter X and y down to those indices
            X_resample, y_resample = X[sample_indices], y[sample_indices]

            # initialize a tree with these resamples num_tree times
            trees[i] = DecisionTree(num_features = num_feats)
            trees[i].fit(X_resample,y_resample)

        return trees

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''

        return np.array([self.root.predict_one(row) for row in X])


    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        return sum(self.predict(X) == y) / float(len(y))
