import inline as inline
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split,StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree,preprocessing
from sklearn.feature_selection import RFECV

def classify(name,clf,Xtrain,ytrain):
    nfolds = 5
    for i in range(0,len(clf)):
        acc = 0
        k_fold = KFold(Xtrain.shape[0], n_folds=nfolds,random_state=0, shuffle=True)
        for train, test in k_fold:
            XTrain, XTest, yTrain, yTest = Xtrain.loc[train], Xtrain.loc[test], ytrain.loc[train], ytrain.loc[test]
            model = clf[i].fit(XTrain, yTrain)
            yPred = model.predict(XTest)
            score = accuracy_score(yTest, yPred, normalize=True)
            acc += score
        print("{0} has an accuracy of: {1}\n".format(name[i],acc/nfolds))

def Test(name, clf, Xtrain, ytrain, Xtest, ytest):
    model = clf.fit(Xtrain,ytrain)
    yPred = model.predict(Xtest)
    score = accuracy_score(ytest, yPred)
    print("Accuracy on test set by {0} is :{1}".format(name,score))

# Read Training data CSV file directly from the desktop and save the results
train_data = pd.read_csv("/Users/Akshita/Desktop/INF550/Adult_Training_63.csv")

# Select the features and targets
feature_number = list(range(62))
XTrain = train_data[feature_number]
YTrain = train_data['label']

# Read Testing data CSV file directly;l
test_data = pd.read_csv("/Users/Akshita/Desktop/INF550/Test_raw_63.csv")
XTest = test_data[feature_number]
YTest = test_data['label']
print(YTest)

# Let's look at how relevant each feature is using extra trees classifier
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=300, criterion = "entropy", random_state=1)
forest.fit(XTrain, YTrain)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.title("Feature importances")
plt.bar(range(XTrain.shape[1]), importances[indices],color="b")
plt.xticks(range(XTrain.shape[1]), XTrain[indices] ,rotation=80, size = 'xx-small', stretch = 'extra-condensed')
plt.xlim([-1 , XTrain.shape[1]])
plt.grid()
plt.show()

# We now look to reduce our feature set to get the core features
estimator = SVC(kernel="linear") #We use a linear SVM as an external estimator
# The "accuracy" attribute being proportional to the correct classification
model = RFECV(estimator, step=1, cv=StratifiedKFold(YTrain, 5),scoring='accuracy')
newModel = model.fit(XTrain, YTrain)

print("Optimal number of features : {0}" .format(newModel.n_features_))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (no. of correct classifications)")
plt.plot(range(1, len(newModel.grid_scores_) + 1), newModel.grid_scores_)
plt.show()

# The graph shows that the accuracy increases only minusculely beyond 40 features
# Let's proceed the experiment with only top 40 features
redFeatNo = list(range(40))
XTrain = XTrain[indices[redFeatNo]]
XTest = XTest[indices[redFeatNo]]
clfs = [GaussianNB(),KNeighborsClassifier()] #Array of classifiers
names = ["Naive Bayes","KNN"]
classify(names,clfs,XTrain,YTrain) # Call the 'classify' function
Test(names[1],clfs[1],XTrain,YTrain,XTest,YTest) # Call to the 'Test' function