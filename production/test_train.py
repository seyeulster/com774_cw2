import unittest
import pytest
import pandas as pd
from model import loadData, splitData, buildModelLR, assessModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class TestTrain(unittest.TestCase):

    def setUp(self):
        test_df = pd.read_csv("./production/testDatasetNew.csv", sep=',')
        self.testX = test_df.drop(['class'], axis=1)
        self.testY = test_df['class']

    def test_loadData(self):
        #The loaded data should have 4944 rows
        #Also, X and Y should be the same length
        X, Y = loadData('./production/actionDatasetNew.csv')
        self.assertGreaterEqual(len(X), 4944)
        self.assertEqual(len(Y), len(X))
        #We The number of columns in X should be equal to the number of features
        #   for the first entry
        self.assertEqual(len(X.iloc[0, :]), 43)

    def test_splitData(self):
        # Test that we can split the data into train and test sets
        # Train and test sets should add up to the same
        # length as the original data (i.e. we havn't lost anything)
        split_pct = 0.3
        X_train, X_test, Y_train, Y_test = splitData(self.testX, self.testY, split_pct)
        self.assertEqual(len(X_train) + len(X_test), len(self.testX))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.testY))
        self.assertEqual(len(X_train), round((1-split_pct)*len(self.testX),0))
        self.assertEqual(len(X_test), round((split_pct)*len(self.testX),0))
        self.assertEqual(len(Y_train), round((1-split_pct)*len(self.testY),0))
        self.assertEqual(len(Y_test), round((split_pct)*len(self.testY)),0)

    def test_buildModel(self):
        #Test the model builder returns a model of the correct type
        model = buildModelLR(self.testX, self.testY)
        self.assertIsInstance(model, LogisticRegression)

    def test_assessModel(self):
        #Test the accuracy function returns a value >=0 and <=1
        #Were giving the same test and train data which we shouldn't
        # but this is a test of its function not it's performance.
        model = buildModelLR(self.testX, self.testY)
        acc = assessModel(model, self.testX, self.testY)
        
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

if __name__ == '__main__':
    unittest.main()