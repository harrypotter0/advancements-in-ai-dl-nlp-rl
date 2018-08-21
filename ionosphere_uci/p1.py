## 1. Import Libraries

import requests
import os 
import pandas as pd 

## Load the datasets
file1 = open("ionosphere.txt","r")
raw_data = file1.read()

## 2. # Processing data into shape

rows = raw_data.strip('\n').split('\n')
rows = [row.split(',') for row in rows]
df = pd.DataFrame(rows)
df.rename(columns={df.columns.values[-1]: 'target'}, inplace=True)
df.to_csv('ionosphere.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cross_validation import KFold, train_test_split
from sklearn.decomposition import PCA

%matplotlib inline
rnd.seed(111)

data = pd.read_csv('ionosphere_processed.csv', index_col=0)
data.drop('1', axis=1, inplace=True)
X = data.iloc[:, :-1].as_matrix()
y = np.array(data.iloc[:, -1])

logit_classifier = LogisticRegression(solver='liblinear')
logit_classifier.fit(X, y)
yhat = logit_classifier.predict(X)
probabilities = logit_classifier.predict_proba(X)

probabilities.sum(axis=1)

conf_mat_0 = confusion_matrix(y_true=y, y_pred=yhat)
conf_mat_0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
                                                    random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(data,label)

predictions=clf.predict(test_data)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,predictions))

result=clf.predict(testdat)

index=[testset['PassengerId']]
df=pd.DataFrame(data=result,index=testset['PassengerId'],columns=['Survived'])
df.to_csv('gender_submission.csv',header=True)
print('gender_submission.csv')
