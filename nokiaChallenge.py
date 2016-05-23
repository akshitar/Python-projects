import numpy as np
from numpy import *
import pandas as pd
import csv

# Define all functions
#******************************************************#
def AssignmentToDF(dataframe, index, dict, column):
    dataframe[column[2]].loc[index] = dict[column[2]]
    dataframe[column[1]].loc[index] = dict[column[1]]
    dataframe[column[0]].loc[index] = dict[column[0]]

def setKey(dictionary, key, defaultVal):
    dictionary.setdefault(key,defaultVal)

#******************End of functions*********************#
column = ['Timestamp', 'userID', 'URL']
rows = 0
with open('/Users/Akshita/Desktop/input.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerow(column)
    with open('/Users/Akshita/Desktop/input.txt') as f:
        for line in f:
            lis = line.split('\r')
            for list in lis:
                s = list.split(',')
                a.writerow([s[0], s[1], s[2]])
                rows += 1

with open('/Users/Akshita/Desktop/input.csv', 'rb') as csvfile:
    data = pd.DataFrame(0, index=np.arange(rows), columns = column)
    reader = csv.DictReader(csvfile)
    for i,row in enumerate(reader):
        AssignmentToDF(data, i, row, column)

user_id = np.unique(data['userID'])
dict_userURL = {}
dict_URLs = {}
for i,key in enumerate(user_id):
    userData = data[data['userID']==key]
    if len(userData)>1:
        setKey(dict_userURL, key, [])
        uniqueURL = np.sort(np.unique(userData['URL']).ravel())
        dict_userURL[key].append((uniqueURL))
        rangeLength = len(dict_userURL[key][0])-1
        for i in xrange(0,rangeLength):
            for num in xrange(i,rangeLength):
                newKey = dict_userURL[key][0][i]+' '+dict_userURL[key][0][num+1]
                setKey(dict_URLs, newKey, 0)
                dict_URLs[newKey] += 1

for key,val in sorted(dict_URLs.items(), key=lambda value: value[1], reverse=True):
    for url in (key.split(' ')):
        print(url),
        print(','),
    print(val)