#we will use iris dataset
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np

#load the dataset
data = load_iris()

model = GaussianNB()
model.fit(data.data, data.target)

#evalaute
print(model.score(data.data, data.target))

#predict
# print(model.predict([4.2, 3, 0.9, 2.1])) #0 = setosa,1 = versicolor, and 2 = virginica

#import the dependencies
from sklearn.datasets import load_iris
from sklearn.svm import SVC
#load dataset
dataset = load_iris()
data = dataset.data
target = dataset.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(data) #check out preprocessing module of sklearn to learn more about preprocessing in ML

#now let us divide data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42, test_size=0.3)

#train a model
model = SVC()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))
model.support_vectors_.shape
model.support_vectors_

#we have already imported libs and dataset
model2= SVC(kernel="rbf", gamma=0.2)
model2.fit(X_train, y_train)
model2.score(X_test, y_test)

# Decision Trees
#entropy in numpy
import numpy as np
def entropy(p):
    return -p * np.log2(p) - (1-p) * np.log2((1-p)) #for binary class

#import libs and dataset
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
#decision trees are prone to overfitting thats why we remove some sub-nodes of the tree, that's called "pruning"
#here, we control depth of the tree using max_depth attribute
#other option for criterion is "gini"
#random_state- just to make sure we get same results each time we run this code

model.fit(X_train, y_train)

#test the model
model.score(X_test, y_test)

from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO() 
export_graphviz(model, out_file=dot_data) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 

from IPython.display import Image
# Image(filename="../images/tree.png")

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
min_estimator = 30  #min number of trees to be built
max_estimator = 60  #max number of trees to be built
rf = RandomForestClassifier(criterion="entropy", warm_start=True, oob_score=True,random_state=42)
for i in range(min_estimator, max_estimator+1):
    rf.set_params(n_estimators=i)
    rf.fit(X,y)   #do not need to seperate training and testing set
    oob_score = 1 - rf.oob_score_
    print(i, oob_score)

#rf.score(X,rf.predict(X)) 

# AdaBoost in sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#load the dataset
dataset = load_iris()

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)
#optioanl - Standardize inputs using StandardScaler
#instantiate a model
model = AdaBoostClassifier(n_estimators=150, random_state=42)
#train a model
model.fit(X_train, y_train)  

print("accuracy: ", model.score(X_test, y_test))

# Gradient Boosting in sklearn
#we use most of the algorithms for classification algorithm, for this lets go for regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#load the dataset
dataset = load_boston()

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)
#optioanl - Standardize inputs using StandardScaler
#instantiate a model
model = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01, random_state=42)
#train a model
model.fit(X_train, y_train)  
from sklearn.metrics import mean_squared_error
print("Mean squared error: ", mean_squared_error(y_test, model.predict(X_test)))

