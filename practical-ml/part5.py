#import dependecies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#read data set
df = pd.read_csv("data/Titanic/train.csv")
df.head()
df.describe()
#prepare data set
X = pd.DataFrame()
X['Pclass'] = df['Pclass']
X['Sex'] = df['Sex']
X['Age'] = df['Age']
X['Survived'] = df['Survived']
X = X.dropna(axis=0)
X.head()
#seperate data and target vars
y = X['Survived'] #don't forget to save target(dependent) var- once we'll drop it we won't be able to get it back
X = X.drop(['Survived'],axis=1)
#let's make sure
X.head()
X['Sex'] = pd.get_dummies(X.Sex)['male'] #1 for male or else 0
scaler = StandardScaler()
X =scaler.fit_transform(X)  #why I need to do this? -> ans - http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
#checking accuracy on training dataset
model.score(X_train, y_train)
pred = model.predict(X_test)
#better metric for binary classification is area under the curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, pred)
print(auc)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
#The f1-score gives you the harmonic mean of precision and recall.
#The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data 
#points in that particular class compared to all other classes.
#The support is the number of samples of the true response that lie in that class.
