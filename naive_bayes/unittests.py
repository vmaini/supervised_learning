'''
Unit tests for Naive Bayes
Usage: from main nlp directory run: make test
'''

import unittest as unittest
import numpy as np
from src.naive_bayes import NaiveBayes

def laplace(n, d, p):
    return (n + 1.) / (d + 1. * p)

class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        X = ['a long document about fishing',
             'a book on fishing',
             'a book on knot-tying']
        self.X = [x.split() for x in X]
        self.y = np.array(['fishing', 'fishing', 'knot-tying'])
        self.nb = NaiveBayes()
        self.nb.fit(self.X, self.y)

    def tearDown(self):
        self.X = None
        self.y = None
        self.nb = None

    def test_class_freq(self):
        self.assertEqual(self.nb.class_freq['fishing'], 2)
        self.assertEqual(self.nb.class_freq['knot-tying'], 1)

    def test_class_counts(self):
        self.assertEqual(self.nb.class_counts['fishing'], 9)

    def test_p_is_number_features(self):
        self.assertEqual(self.nb.p, 8)

    def test_class_feature_counts(self):
        self.assertEqual(self.nb.class_feature_counts['knot-tying']['fishing'], 0)
        self.assertEqual(self.nb.class_feature_counts['fishing']['document'], 1)
        self.assertEqual(self.nb.class_feature_counts['fishing']['fishing'], 2)

    def test_predict(self):
        test_X = [["book"]]
        p = 8
        fishing_likelihood = sum((np.log(laplace(1, 9, p)),np.log(2/3.)))
        knot_tying_likelihood = sum((np.log(laplace(1, 4, p)),np.log(1/3.)))
        posts = self.nb.posteriors(test_X)
        preds = self.nb.predict(test_X)

        self.assertEqual(fishing_likelihood, posts[0]['fishing'])
        self.assertEqual(knot_tying_likelihood, posts[0]['knot-tying'])
        self.assertEqual(preds[0], 'fishing')
        self.assertNotEqual(preds[0], 'knot-tying')

    def test_score(self):
        self.assertEqual(self.nb.score(self.X, self.y), 1.)


if __name__ == '__main__':
    unittest.main()
