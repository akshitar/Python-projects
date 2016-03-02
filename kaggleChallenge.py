import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from patsy import dmatrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2

np.set_printoptions(precision=5, threshold=np.inf)

# Function for evaluating new set of features for different thresholds of chi2 values
def chi2Dataframe(threshold,dataset,chiValues):
    newDataset = pd.DataFrame()
    for i in range(0,chiValues.shape[0]):
        if chiValues[i]>threshold:
            newDataset[i]=dataset[i+1]
        else: continue
    return newDataset

# Read Training data CSV file directly from the desktop and save the results
result = pd.read_csv("/home/ubuntu/walmartdata(woV).csv").dropna(axis  = 1, how = 'any')

#Giving asymmetrical class numbers symmetry
result['TripType'] = result['TripType'].convert_objects(convert_numeric=True)
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
    dataset[i] = dataset[i].convert_objects(convert_numeric=True)

# Caluculate the chi2 values
chiValues,pval = chi2(dataset[feature_number],dataset[0])
threshold = 300
newDataset = chi2Dataframe(threshold, dataset,chiValues)

zeroCol = pd.DataFrame(dataset[0])
newDataset = pd.concat([zeroCol , newDataset] , join='outer', axis =1)
newDataset.columns = arange(0,len(newDataset.columns))
split = np.random.permutation(int(rows*0.9))
redDataset = newDataset.iloc[split]
rows,cols = redDataset.shape
feature_number = list(range(1,cols))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(redDataset[feature_number], redDataset[0], test_size=0.3,random_state=1)

clf = GridSearchCV(LinearSVC(penalty='l2',dual=True), cv=5,param_grid={"C": [0.001,0.1,1e0, 1e1, 1e2, 1e3]})#,"gamma": np.logspace(-2, 2, 5)})
clf.fit(X_train,y_train)
yTest_pred = clf.predict(X_test)
yTrain_pred = clf.predict(X_train)
print("Score method for validation with Grid CV for 0.9: {0}".format(accuracy_score(y_test, yTest_pred)))
print("Score method for training with Grid CV: {0}".format(accuracy_score(y_train, yTrain_pred)))
print("All done")

"""for rows*0.5: Score method for validation with Grid CV: 0.354002730047
    Score method for training with Grid CV: 0.364098593367"""
"""for rows*0.7 : Score method for validation with Grid CV: 0.354520948639
    Score method for training with Grid CV: 0.362271015"""