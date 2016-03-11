import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import random
from sklearn import cross_validation
from sklearn import svm
from patsy import dmatrix
from sklearn.utils import resample
from sklearn.svm import NuSVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2

np.set_printoptions(precision=5, threshold=np.inf)
pd.set_option('max_columns', 80)

"""# Function for evaluating new set of features for different thresholds of chi2 values
def chi2Dataframe(threshold,dataset,chiValues):
    newDataset = pd.DataFrame()
    for i in range(0,chiValues.shape[0]):
        if chiValues[i]>threshold:
            newDataset[i]=dataset[i+1]
        else: continue
    return newDataset"""

# Read Training data CSV file directly from the desktop and save the results
result = pd.read_csv("/home/ubuntu/walmartdata(woV).csv").dropna(axis  = 1, how = 'any')
del result['Upc']

#Giving asymmetrical class numbers symmetry
result['TripType'] = result['TripType'].convert_objects(convert_numeric=True)
clsNum = np.sort(pd.unique(result['TripType'].ravel()))
j = 1
for i in clsNum:
    result['TripType'].loc[result['TripType']==i] = j
    j +=1
a = np.sort(pd.unique(result['TripType'].ravel()))
#print('Sorted new class numbers: {0}'.format(a))

#Count number of samples in each class
nSamples = []
for i in range(1,len(clsNum)+1):
    nSamples.append(sum(result['TripType']==i))

print("Min number of samples: {0} and for class: {1}".format(min(nSamples),(nSamples.index(min(nSamples)))+1))
print("Max number of samples: {0} and for class: {1}".format(max(nSamples),(nSamples.index(max(nSamples)))+1))
print(nSamples)
#Create an empty dataframe which will have over-samples samples
newDataframe = pd.DataFrame(columns=result.columns)

#We need to over sample our minority classes such that each class will have samples=4000

for n,item in enumerate(nSamples):
    if (item<4000):
        dummy = result.loc[result['TripType']==n+1]
        k = len(dummy)
        dummy[:][73] = 0
        num = np.random.permutation(k)
        dummy.insert(73,'randNum',num)
        dummy = dummy.set_index(['randNum'])
        for i in range(0,4000):
            j = random.randrange(0, k, 1)
            newDataframe.loc[len(newDataframe)] = dummy.loc[j]
    else:
        dummy = result.loc[result['TripType']==n+1]
        frames = [newDataframe , dummy]
        newDataframe = pd.concat(frames, ignore_index= True, join  ='outer')

print(newDataframe.shape)
result = newDataframe

# Expand 'ScanCount' and 'FinelineNumber' categories
scanCount = dmatrix('C(ScanCount)-1',result, return_type='dataframe')
flNum = dmatrix('C(FinelineNumber)-1',result, return_type='dataframe')
result = result.drop(['ScanCount', 'FinelineNumber'], axis=1)

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
split = np.random.permutation(int(rows*0.5))
redDataset = newDataset.iloc[split]
rows,cols = redDataset.shape
feature_number = list(range(1,cols))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(redDataset[feature_number], redDataset[0], test_size=0.3,random_state=1)

rbf_clf = GridSearchCV(NuSVC(kernel='rbf'), cv=5,param_grid={"C": [0.001,0.1,1e0, 1e1, 1e2, 1e3],"gamma": np.logspace(-2, 2, 5)})
rbf_clf.fit(X_train,y_train)
yTest_pred = rbf_clf.predict(X_test)
yTrain_pred = rbf_clf.predict(X_train)
print("Score method for validation with RBF kernel for 0.9: {0}".format(accuracy_score(y_test, yTest_pred)))
print("Score method for training with RBF kernel: {0}".format(accuracy_score(y_train, yTrain_pred)))
print("All done")

"""for rows*0.5: Score method for validation with Grid CV: 0.354002730047
    Score method for training with Grid CV: 0.364098593367"""
"""for rows*0.7 : Score method for validation with Grid CV: 0.354520948639
    Score method for training with Grid CV: 0.362271015"""
""" for rows*0.9: Score method for validation with Grid CV for 0.9: 0.35639247
    Score method for training with Grid CV: 0.362732586761"""
"""for rbf 0.3: Score method for validation with Grid CV for 0.9: 0.360083864924
    Score method for training with Grid CV: 0.360572212424"""
    