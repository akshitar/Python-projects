import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Read Training data CSV file directly from the desktop and save the results
result = pd.read_csv("/Users/Akshita/Desktop/Kaggle_walmart/walmartdata(woV).csv").dropna(axis  = 1, how = 'any')
start_time = datetime.now()

# Expand 'ScanCount' and 'FinelineNumber' categories
scanCount = dmatrix('C(ScanCount)-1',result, return_type='dataframe')
flNum = dmatrix('C(FinelineNumber)-1',result, return_type='dataframe')

#Concatenate the resulting expanded features
dataset = pd.concat([result, scanCount, flNum], axis=1)
dataset = dataset.drop(['ScanCount', 'FinelineNumber','Upc'], axis=1)
dataset.columns = arange(0,len(dataset.columns))

#Giving asymmetrical class numbers symmetry
dataset[0] = pd.to_numeric(dataset[0])
clsNum = np.sort(pd.unique(dataset[0].ravel()))
j = 1
for i in clsNum:
    dataset[0].loc[dataset[0]==i] = j
    j +=1

rows,cols = dataset.shape
print("The number of samples= {0}\nThe number of features= {1}".format(rows,cols-1))
feature_number = list(range(1,422))

# Select k best features
#clf = RandomForestClassifier(n_estimators = 20)
nFeatures = np.array([100])#, 3000, 1000, 500, 100])
nfolds = 2
for i in range(0, len(nFeatures)):
    featSelect = SelectKBest(chi2 , k = nFeatures[i])
    X_kbest = featSelect.fit_transform(dataset[feature_number], dataset[0])
    print(X_kbest)

print('Duration: {0}'.format(datetime.now()- start_time))
"""kfold = KFold(rows, n_folds = nfolds)
    clf = RandomForestClassifier(n_estimators = 20)
    for train_indx, test_indx in kfold:
        XTrain, XTest, yTrain, yTest = X_kbest[test_indx], X_kbest[train_indx], X_kbest[test_indx], X_kbest[train_indx]
        clf.fit(XTrain, yTrain)
        yPred = clf.predict(XTest)
        score = accuracy_score(yTest, yPred)"""
