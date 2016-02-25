import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from patsy import dmatrix
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,train_test_split

# Function for evaluating new set of features for different thresholds of chi2 values
def chi2Dataframe(threshold,dataset,chiValues):
    newDataset = pd.DataFrame()
    for i in range(0,chiValues.shape[0]):
        if chiValues[i]>threshold:
            newDataset[i]=dataset[i+1]
        else: continue
    return newDataset

np.set_printoptions(precision=5, threshold=np.inf)

# Read Training data CSV file directly from the desktop and save the results
result = pd.read_csv("/home/ubuntu/walmartdata(woV).csv").dropna(axis  = 1, how = 'any')

#Giving asymmetrical class numbers symmetry
result['TripType'] = pd.to_numeric(result['TripType'])
clsNum = np.sort(pd.unique(result['TripType'].ravel()))

j = 1
for i in clsNum:
    result['TripType'].loc[result['TripType']==i] = j
    j +=1

# Expand 'ScanCount' and 'FinelineNumber' categories
scanCount = dmatrix('C(ScanCount)-1',result, return_type='dataframe')
flNum = dmatrix('C(FinelineNumber)-1',result, return_type='dataframe')
result = result.drop(['ScanCount', 'FinelineNumber','Upc'], axis=1)

#Concatenate the resulting expanded features
dataset = pd.concat([result, scanCount, flNum], axis=1)
dataset.columns = arange(0,len(dataset.columns))
rows,cols = dataset.shape
print("The number of samples= {0}\nThe number of features= {1}".format(rows,cols-1))
feature_number = list(range(1,cols))
dataset = dataset.replace(' ', 0)
for i in range(0, cols):
    dataset[i] = pd.to_numeric(dataset[i])

# Caluculate the chi2 values
chiValues,pval = chi2(dataset[feature_number],dataset[0])
threshold = 300
newDataset = chi2Dataframe(threshold, dataset,chiValues)

Cs = np.linspace(-5, 15, num=11, dtype=float)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(newDataset, dataset[0], test_size=0.3,random_state=1)
for i in range(0,len(Cs)):
    clf = svm.SVC(kernel='linear', C=(2**Cs[i])).fit(X_train, y_train)
    print clf.score(X_test, y_test)