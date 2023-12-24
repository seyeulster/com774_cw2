import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Load data from a CSV using pandas and return it.
# expected format is two columns.
def loadData(file):
    # Load the CSV file
    df = pd.read_csv(file, sep=";")
	# drop from df missing/NaN values using dropna() function 
    df = df.dropna()
    # Pick out the columns we want to use as inputs
    X = df.drop(['id', 'action', 'video_src'], axis=1)
    # Pick out the column we want to use as the output
    Y = df['action']
    # Return the input and output
    return X, Y

# Split the data into train and test sets
def splitData(X, Y, percentage=0.3):
	#Pretty straightforward, use the scikit-learn function
	return train_test_split(X, Y, test_size=percentage)

def buildModelLR(X, Y):
	#Create a model and fit it to the data
	mdl = LogisticRegression(C=1/0.1, solver="liblinear", random_state=42)
	mdl.fit(X, Y)
	return mdl
def buildModelSVM(X, Y):
	mdl = SVC()
	mdl.fit(X, Y)
	return mdl

def assessModel(model, X, Y):
	#Get the predictions for the test data and compute accuracy
	testPredictions = model.predict(X)
	acc = np.average(testPredictions == Y)
	return acc

def trainModel(dataFile, modelSavePath, modelType='LR'): 
	X, Y = loadData(dataFile)
	X_train, X_test, Y_train, Y_test = splitData(X, Y, 0.2)
	print("Train length", len(X_train))
	print("Test length", len(X_test))
	if modelType == "LR":
		model = buildModelLR(X_train, Y_train)

	acc = assessModel(model, X_train, Y_train)
	print("Accuracy", acc)
	if modelSavePath:
		print("Saving model to", modelSavePath)
		mlflow.sklearn.save_model(model, modelSavePath)
	return model, X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--trainingdata', required=True, help='Data file to load')
	argparser.add_argument('--model', default=False, help='Model file to save')
	args = argparser.parse_args()
	#Let's use autologging because it's awsome.
	mlflow.autolog()
	trainModel(args.trainingdata, args.model)
