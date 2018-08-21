#sklearn comes with few small datasets. We will use one of them called "boston housing". Which is identical to
#to the example we saw in theory part. This dataset has 506 samples with 13 features (columns). Here target variable
#is the price of the house.

#1.import the libs
from sklearn.datasets import load_boston
#load the dataset
data = load_boston()  #returns dictionary-like object, attributes are - data, target, DESCR
#first of all, let's see the shape of the training data
print(data.data.shape)
#shape of a target/labels
print(data.target.shape)
#important info about the dataset
print(data.DESCR)
#how target values look like
data.target[:40]


#2. Train the model
from sklearn.linear_model import LinearRegression
#create a linear regression object
lin_reg = LinearRegression()
#train a model
lin_reg.fit(data.data, data.target)
#learned weights
lin_reg.coef_
#learned intercept
lin_reg.intercept_

#3. Test a model
# we can use a model to predict as follows
lin_reg.predict(data.data[4].reshape(1,-1))  #first sample
#let's see what was the true value
data.target[4]  #not good :(
#find mean squared error
from sklearn.metrics import mean_squared_error
mean_squared_error(data.target, lin_reg.predict(data.data))
#let us calculate mse from scratch to make sure its correct
import numpy as np
np.mean((lin_reg.predict(data.data) - data.target) ** 2)

# 5. Deploy a model 
# Logistic regression
def sigmoid(x):
    return 1/(1+np.exp(x))
import numpy as np
numbers = np.linspace(-20,20,50) #generate a list of numbers
numbers
#we will pass each number through sigmoid function
results = sigmoid(numbers)
results[:10]  #print few numbers

# Now, we will implement logistic regression using sklearn.
#this time we will use digit dataset.
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
# %matplotlib inline

digits = load_digits()
X = digits.data  #input
y = digits.target #output
print(digits.data.shape)  #1797 samples * 64 (8*8)pixels
#input is an image and we would like to train a model which can predict the digit that image contains
#each image is of 8 * 8 pixels


#plot few digits  ## dont worry if u dont understand it
# 1. Load the dataset
fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(231)
ax1.imshow(digits.images[0])

ax2 = fig.add_subplot(232)
ax2.imshow(digits.images[1])

ax3 = fig.add_subplot(233)
ax3.imshow(digits.images[2])

# plt.tight_layout()
# plt.show()

# 2. Train the model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
#train a model
log_reg.fit(X, y)

# 3. Test a model
# sklearn provides several ways to test a classifier
from sklearn.metrics import accuracy_score
accuracy_score(y, log_reg.predict(X))
#another way
log_reg.score( X, y)
#confusion matrix is a table that can be used to evaluate the performance of a classifier
#each row shows actual values and column values shows predicted values
from sklearn.metrics import confusion_matrix
confusion_matrix(y, log_reg.predict(X))

# 4. Deploy the model
#we can use predict method to predict the class
print("Predicted : " , log_reg.predict(digits.data[1].reshape(1,-1)))
print("Actual : ", digits.target[1])

#we can also predict the probability of each class
proba = log_reg.predict_proba(digits.data[1].reshape(1,-1)) # second column has the highest probability
print(proba)
np.argmax(proba) #please note index starts with 0

from sklearn.datasets import load_iris #https://en.wikipedia.org/wiki/Iris_flower_data_set
iris = load_iris()
iris.data.shape 

#split the dataset into two parts
from sklearn.model_selection import train_test_split
#split dataset into 70-30
X_train, X_test, y_train , y_test = train_test_split(iris.data, iris.target, test_size= 0.3, random_state=42)
#randomstate - to make sure each time we run this code it gives same results

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#train on training data
model = LogisticRegression()
model.fit(X_train, y_train)
#test on test data
model.score(X_test, y_test)

# Regularization
model = LogisticRegression(penalty="l2", C=1) #default configuration
model.fit(X_train, y_train)
model.score(X_test, y_test) #note, we got same accuracy

#let us use l1 regularization
model = LogisticRegression(penalty="l1", C=1)
model.fit(X_train, y_train)
model.score(X_test, y_test) #whoa! we got 100% accuracy

model = LogisticRegression(penalty="l2", C=0.23)
model.fit(X_train, y_train)
model.score(X_test, y_test) 

#you have to consider various values for this type of parameters (hyperparameters) to find the best one
# we can do this with GridCV, RandomCV- This is beyond the scope of this lab session

# Cross-validation
#we discussed about k-fold cross validation. In which we divide whole dataset into k parts and each time we hold out
#one part and train on k-1 parts

#we'll use boston housing dataset
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5) #k=10

costs = []
for train_index,test_index in kfold.split(data.data):
    X_train, y_train = data.data[train_index], data.target[train_index]
    X_test, y_test = data.data[test_index], data.target[test_index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    costs.append(mean_squared_error(y_test, model.predict(X_test)))


np.mean(costs)

#10 fold cross-validation
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
digits = load_digits()


model = LogisticRegression()
scores = cross_val_score(model,digits.data, digits.target, cv=10, scoring='accuracy' )
scores.mean()

from sklearn.model_selection import StratifiedKFold
digits = load_digits()
skfold = StratifiedKFold(n_splits= 10)
costs = []
for train_index,test_index in skfold.split(digits.data, digits.target):
    X_train, y_train = digits.data[train_index], digits.target[train_index]
    X_test, y_test = digits.data[test_index], digits.target[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    costs.append(model.score(X_test, y_test))

np.mean(costs)

