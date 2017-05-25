from adaboost import AdaBoostBinaryClassifier
import numpy as np
from sklearn.model_selection import train_test_split

if __name__=='__main__':
   data = np.genfromtxt('../data/spam.csv', delimiter=',')

   y = data[:, -1]
   x = data[:, 0:-1]

   train_x, test_x, train_y, test_y = train_test_split(x, y)

   my_ada = AdaBoostBinaryClassifier(n_estimators=50)
   my_ada.fit(train_x, train_y)
   print "Accuracy:", my_ada.score(test_x, test_y)
