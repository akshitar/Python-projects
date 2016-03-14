import re
import csv
import json
import pandas as pd
import numpy as np
from numpy import unique,array
from sklearn.metrics import jaccard_similarity_score
from sklearn.cluster import AgglomerativeClustering

def compute_jaccard_index(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)

def cleanArr(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i].replace('token_', '')
        arr[i] = arr[i].strip()
    return arr

def bar(x,columns):
    return ','.join(list(columns[x]))

count = 0
productFeatures = dict()
with open('data/skuFeatures.csv', 'rb') as features:
    reader = csv.reader(features)
    for row in reader:
        if count != 0:
            productFeatures[row[0]] = cleanArr(row[1].split(' '))
        count = count + 1
print('Found all the features')

count = -1
maxCount = 0
userFeatures = dict()
userProducts = dict()
with open('data/transactionLog.csv', 'rb') as transactionLog:
    reader = csv.reader(transactionLog)
    for userData in reader:
        count = count + 1
        if (count == 0):
            continue
        if count % 100 == 0:
            print(count)
            break
        if maxCount < userData[0]:
            maxCount = int(userData[0])
        if str(userData[1]) in productFeatures:
            # Append product features to the user profile
            if str(userData[0]) in userFeatures:
                userFeatures[str(userData[0])] = list(set(userFeatures[str(userData[0])] + productFeatures[str(userData[1])]))
            else:
                userFeatures[str(userData[0])] = productFeatures[str(userData[1])]

            # Append products to the user profile
            if str(userData[0]) in userProducts:
                userProducts[str(userData[0])].append(userData[1])
            else:
                userProducts[str(userData[0])] = [userData[1]]

userFeatureList = np.zeros([maxCount, 0]).tolist()
for key in userFeatures:
    userFeatureList[int(key) - 1] = userFeatureList[int(key) - 1] + userFeatures[key]

jaccardSimilarity = np.matrix(np.zeros([maxCount, maxCount]))
for i in range(0, maxCount):
    for j in range(0, i + 1):
        firstArr = userFeatureList[i]
        secondArr = userFeatureList[j]
        if i == j:
            score = 1
        elif len(firstArr) == 0 or len(secondArr) == 0:
            score = 0
        else:
            score = compute_jaccard_index(set(firstArr), set(secondArr))
        jaccardSimilarity[i, j] = score
        jaccardSimilarity[j, i] = score

# Clustering
nClusters = 5;
clusterList = np.zeros([nClusters, 0]).tolist()
itemsForUsersInCluster = np.zeros([nClusters, 0]).tolist()
itemsInCluster = np.zeros([nClusters, 0]).tolist()

clusters = AgglomerativeClustering(nClusters, connectivity = None, linkage = 'ward').fit_predict(jaccardSimilarity) # should return an array with clusters
for i in range(0, len(clusters)):
    clusterList[clusters[i]].append(i + 1)
    if str(i + 1) in userProducts:
        if len(itemsForUsersInCluster[clusters[i]]) == 0:
            itemsForUsersInCluster[clusters[i]].append(str(i + 1))
        itemsForUsersInCluster[clusters[i]].append(userProducts[str(i + 1)])
        itemsInCluster[clusters[i]] = list(set(itemsInCluster[clusters[i]] + userProducts[str(i + 1)]))


# Make an association rule data table
for clusterNum,item in enumerate(itemsForUsersInCluster):
    columnProducts = unique(itemsInCluster[clusterNum])
    print(columnProducts)
    associationRule = pd.DataFrame(0, index=columnProducts, columns=columnProducts)
    print(associationRule)
    for j,products in enumerate(item):
        if len(products)<=2:
            continue
        else:
            length = len(products)
            for p in range(1,length):
                for q in range(p,length):
                    print(p)
                    print(q)
                    associationRule.loc[products[p], products[q]] += 1
                    associationRule.loc[products[q], products[p]] += 1
        print("For users in  cluster {0} the recommendations are:".format(clusterNum))
        print("For user {0}".format(products[0]))
        ar_boolean = associationRule>0
        ar_boolean['final'] = ar_boolean.apply(lambda x: bar(associationRule,ar_boolean.columns),axis=1)
        ar_output =  ar_boolean[ar_boolean['final'] != '']['final']
        ar_output =  unique(ar_output.ravel())

    print(associationRule)
    break
