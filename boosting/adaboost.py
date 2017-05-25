import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        for i in xrange(self.n_estimator):
            boosted = self._boost(x, y, self.estimator_weight_)
            self.estimators.append(boosted[0])
            self.estimator_weight_[i] = boosted[2]

    def _boost(self, x, y, sample_weight):
        '''
        Go through one iteration of the AdaBoost algorithm. Build one estimator.

        Returns:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)
        '''

        estimator = clone(self.base_estimator)
        estimator.fit(x, y, sample_weight=sample_weight)

        estimator_error = np.sum(sample_weight * (y != estimator.predict(x))) / np.sum(sample_weight)
        estimator_weight = np.log2((1 - estimator_error) / estimator_error)
        sample_weight = sample_weight * np.exp(estimator_weight * (y != estimator.predict(x)))

        #update thes sample weights
        self.sample_weight = sample_weight

        return estimator, sample_weight, estimator_weight

        # Calculate the error term (estimator_error)
        # Calculate the alpha (estimator_weight)
        # Update the weights (sample_weight)


    def predict(self, x):
        '''
        Takes feature matrix and returns numpy array of 0/1 labels
        '''

        # creates a list of predictions for each estimator, with each prediction translated to a -1 or 1
        predictions = [[lambda x: 1 if x == 1 else -1 for x in prediction] for prediction in [estimator.predict(x) for estimator in self.estimators_]]

        weighted = np.array([self.estimator_weight_ * preds for preds in predictions])
        final_predictions = np.sum(weighted,axis=0)

        return [0 if p < 0 else 1 for p in final_predictions]

        # final_predictions = [1] * len(predictions)
        # for idx, p in enumerate(predictions):
        #     for i in p:
        #         final_predictions *= p[i] * self.estimator_weight_[idx]

        # alternative if above crazy double-nested list comprehension doesn't work...
        # predictions = [estimator.predict(x) for estimator in self.estimators_]
        # for prediction in predictions:
        #      negatized_predictions.append([lambda x: 1 if True else -1 for p in prediction])


    def score(self, x, y):
        return np.mean(y == self.predict(x))
