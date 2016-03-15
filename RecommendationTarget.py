import re
import csv
import json
import pandas as pd
import numpy as np
from numpy import unique,array
from sklearn.metrics import jaccard_similarity_score
from sklearn.cluster import AgglomerativeClustering

###################
#### Variables ####
###################
count = 0 # iterator
highestUserID = 0 # stores highest uid among the users
userProducts = dict() # stores products bought for each user
userFeatures = dict() # stores features of each user based on purchased products
productFeatures = dict() # stores features for each product

###################
#### Functions ####
###################
def computeJaccardIndex(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)

def cleanArr(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i].replace('token_', '')
        arr[i] = arr[i].strip()
    return arr

def bar(x,columns):
    return ','.join(list(columns[x]))

def loadSKUFeatures(filePath):
    count = 0
    with open(filePath, 'rb') as features:
        reader = csv.reader(features)
        for row in reader:
            if count != 0:
                productFeatures[row[0]] = cleanArr(row[1].split(' '))
            count = count + 1
    print('Found all the features')

def loadTransactionLog(filePath, highestUserID):
    with open(filePath, 'rb') as transactionLog:
        count = -1
        reader = csv.reader(transactionLog)
        for userData in reader:
            count = count + 1
            if (count == 0):
                continue
            if count % 10000 == 0:
                break

            uID = str(userData[0])
            skuID = str(userData[1])

            # Get the highest userID
            if highestUserID < int(uID):
                highestUserID = int(uID)

            if skuID in productFeatures:
                # Append product features to the user profile
                if uID in userFeatures:
                    userFeatures[uID] = list(set(userFeatures[uID] + productFeatures[skuID]))
                else:
                    userFeatures[uID] = productFeatures[skuID]

                # Append products to the user profile
                if uID in userProducts:
                    userProducts[uID].append(skuID)
                else:
                    userProducts[uID] = [skuID]
    return highestUserID

def calculateJaccardSimilarityArr(maxUserID, featureList):
    similarityArr = np.matrix(np.zeros([maxUserID, maxUserID]))
    for i in range(0, maxUserID):
        for j in range(0, i + 1):
            firstArr = featureList[i]
            secondArr = featureList[j]
            if i == j:
                score = 1
            elif len(firstArr) == 0 or len(secondArr) == 0:
                score = 0
            else:
                score = computeJaccardIndex(set(firstArr), set(secondArr))
            similarityArr[i, j] = score
            similarityArr[j, i] = score
    return similarityArr

def featureBasedClustering(count, similarityMetrics):
    return AgglomerativeClustering(count, connectivity = None, linkage = 'ward').fit_predict(similarityMetrics)

###################
#### Load Data ####
###################
# Load all features for each product into 'productFeatures'
loadSKUFeatures('data/skuFeatures.csv')

# Load transaction log
highestUserID = loadTransactionLog('data/transactionLog.csv', highestUserID)
print('Status: [#----] Features Loaded')

# Convert userFeatures dict to list
userFeatureList = np.zeros([highestUserID, 0]).tolist()
for key in userFeatures:
    userFeatureList[int(key) - 1] = userFeatureList[int(key) - 1] + userFeatures[key]

#########################
#### User Similarity ####
#########################
# Calculate jaccard similarity between all users
jaccardSimilarity = calculateJaccardSimilarityArr(highestUserID, userFeatureList)
print('Status: [##---] Jaccard Similarity calculated')

####################
#### Clustering ####
####################
clusterCount = 3; # Number of clusters

# Initialize list of users in each cluster
# Separate list denotes separate clusters
clusterList = np.zeros([clusterCount, 0]).tolist()

# Initialize list of users in each cluster
# Separate list denotes separate clusters
clusterUserItems = np.zeros([clusterCount, 0]).tolist()

# Initialize list of items in each cluster
# Separate list denotes separate clusters
clusterItems = np.zeros([clusterCount, 0]).tolist()

# Agglomerative clustering (sklearn)
userGroups = featureBasedClustering(clusterCount, jaccardSimilarity) # should return an array with clusters
print('Status: [###--] User clusters formed')

# Segregate users into their respective clusters
# Also computes clusterItems & clusterUserItems
for i in range(0, len(userGroups)):
    userID = str(i + 1)
    clusterList[userGroups[i]].append(userID)
    if userID in userProducts:
        clusterUserItems[userGroups[i]].append([userID] + userProducts[userID])
        clusterItems[userGroups[i]] = list(set(clusterItems[userGroups[i]] + userProducts[userID]))

# Make an association rule data table
for clusterNum,item in enumerate(clusterUserItems):
    columnProducts = unique(clusterItems[clusterNum])
    associationRule = pd.DataFrame(0, index=columnProducts, columns=columnProducts)
    for j,eachUser in enumerate(item):
        if len(eachUser) <= 2:
            continue
        else:
            length = len(eachUser)
            for p in range(1,length):
                for q in range(p,length):
                    associationRule.loc[eachUser[p], eachUser[q]] += 1
                    associationRule.loc[eachUser[q], eachUser[p]] += 1
    print("For users in  cluster {0} the recommendations are:".format(clusterNum+1))
    for j,eachUser in enumerate(item):
        recommendationDict = dict()
        print("For user {0}".format(eachUser[0]))
        for elemIndex in range(1, len(eachUser)):
            ar_boolean = associationRule.loc[eachUser[elemIndex]] > 0
            ar_boolean = ar_boolean[ar_boolean == True]
            associatedProducts = ar_boolean.index.values
            for i in range(0, len(associatedProducts)):
                if associatedProducts[i] in recommendationDict.keys():
                    recommendationDict[associatedProducts[i]] = recommendationDict[associatedProducts[i]] + associationRule[eachUser[1]][associatedProducts[i]]
                else:
                    recommendationDict[associatedProducts[i]] = associationRule[eachUser[elemIndex]][associatedProducts[i]]
        for Key in sorted(recommendationDict):
            if Key not in eachUser:
                print Key

print('Status: [####-] Association rules formed')
