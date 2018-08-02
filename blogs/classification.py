#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv('data/database.csv')

from sklearn.preprocessing import Imputer

X=Data.iloc[:,[2,3,4,6,7,8,9,10,11,13,14]].values
y=Data.iloc[:,[0,1,5,12]].values


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

Data = pd.read_csv('data/database.csv')
print(Data.info())
print(Data.head())

data =Data
# object_cols = ['PartModel','PartType','InGrdPin','OutGrdPin']
# for da in object_cols:
#     print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
total_cols = data.columns
print("Printing Before")
for da in total_cols:
    print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
for da in total_cols:
    data[da].fillna(0,inplace = True)
print("Printing After")
for da in total_cols:
    print(data[da].value_counts(dropna =False))  # if there are nan values that also be counted
print(data.info())

for da in total_cols:
    data[da] = le.fit_transform(data[da].astype('str'))

print(data.info())



fit = data
XX = fit.iloc[:,1:].values   
yy = fit.iloc[:,1].values



from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(XX,yy,test_size = 0.2, random_state = 0)

# feature Scaling
from  sklearn.preprocessing import StandardScaler

SC_X = StandardScaler()

X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)


# Fitting Decision Tree  Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 43)
classifier.fit(X_train, y_train)

#Predicting result
y_pred = classifier.predict(X_test)

#Checking Accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
#print(accuracy_score(y_test, y_pred, normalize=False))

from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


'''
mean_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
mean_imputer=mean_imputer.fit(X)

imputed_df = mean_imputer.transform(df.values)

'''
##Missing Value Treatment
#
#
#
#from sklearn.preprocessing import Imputer  #god 
#Imp = Imputer(missing_values="NaN", strategy="mean", axis=0) #-- Gods template for Dance
#Imp = Imp.fit(y) #   - Human
#X[:,1:] = Imp.transform(X[:,1:]) # - Ram
#
#
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encoder_x=LabelEncoder()
#y[:,0]=encoder_x.fit_transform(y[:,0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#y=onehotencoder.fit_transform(y).toarray()
#y=[:,1:]
#from sklearn.base import TransformerMixin
#
#class DataFrameImputer(TransformerMixin):
#    def __init__(self):
#        """Impute missing values.
#        """
#    def fit(self, X, y=None):
#        self.fill = pd.Series([X[c].value_counts().index[0]
#            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
#            index=X.columns)
#
#        return self
#
#    def transform(self, X, y=None):
#        return X.fillna(self.fill)
#

#X = pd.DataFrame(Data)
#xt = DataFrameImputer().fit_transform(X)
#
#print('before...')
#print(X)
#print('after...')
#print(xt)

#from collections import defaultdict
#from sklearn.preprocessing import LabelEncoder
#d = defaultdict(LabelEncoder)
#
## Encoding the variable
#fit = xt.apply(lambda x: d[x.name].fit_transform(x))



#from sklearn.externals import joblib
#joblib.dump(classifier,"classifier.pkl")







