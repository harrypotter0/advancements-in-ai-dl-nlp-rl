
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data = pd.read_csv('data.csv')
print data.info()
print data.head()

# object_cols = ['PartModel','PartType','InGrdPin','OutGrdPin']
# for da in object_cols:
#     print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
total_cols = data.columns
print "Printing Before"
for da in total_cols:
    print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
for da in total_cols:
    data[da].fillna(0,inplace = True)
print "Printing After"
for da in total_cols:
    print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
print data.info()

for da in total_cols:
    data[da] = le.fit_transform(data[da].astype('str'))

print data.info()
print data.head(10)
data.to_csv("new_data.csv")