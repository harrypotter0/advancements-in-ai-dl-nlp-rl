# Load the dataset
from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target

# Perform preprocessing
#standardize the data to make sure each feature contributes equally to the distance
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_processed = ss.fit_transform(X)

#split the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Fit the dataset
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric="minkowski", p=2) #p=2 for euclidian distance
model.fit(X_train, y_train)
model.score(X_test, y_test)

# KNN
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

digits = load_digits()
dataset = digits.data

#standardize
ss = StandardScaler()
dataset = ss.fit_transform(dataset)

model = KMeans(n_clusters= 10, init="k-means++", n_init=10)
model.fit(dataset)
model.labels_   #assigned label(cluster) to each data point
model.inertia_  #sum of distances of samples to their closest centroid
model.cluster_centers_

import matplotlib.pyplot as plt
# %matplotlib inline
plt.imshow(digits.images[1], cmap='gray')
model.predict(dataset[1].reshape(1,-1)) #should be one
#lets try again
plt.imshow(digits.images[333], cmap='gray')
model.predict(dataset[333].reshape(1,-1))  #should be 2

# PCA
from sklearn.decomposition import PCA
import numpy as np

#lets create features
x1 = np.random.normal(size=200) 
x2 = np.random.normal(size=200)
x3 = x1 + x2  #not useful since its highly correlated with other features.
X = np.c_[x1,x2,x3]

pca = PCA()
pca.fit(X)
pca.explained_variance_   #third feature is clearly useless
pca.n_components_  #still 3, because we have not specify no of components to keep in PCA() method
pca2 = PCA(n_components=2)
pca2.fit(X)
pca2.n_components_
X_processed = pca2.fit_transform(X)
X.shape
X_processed.shape

# NN:
#read the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#create placeholders to store input and output data
import tensorflow as tf
X = tf.placeholder(tf.float32, shape=[None, 784])  #28* 28 = 784
y = tf.placeholder(tf.float32, shape=[None, 10])  #10 classes
#create weights and bias
w1 = tf.Variable(tf.truncated_normal([784, 50], stddev=0.5))
b1 = tf.Variable(tf.ones([50]))
#for hidden to output layer
w2= tf.Variable(tf.truncated_normal([50,10], stddev=0.5))
b2= tf.Variable(tf.ones([10]))
h = tf.nn.relu(tf.matmul(X,w1)+b1)
o = tf.nn.relu(tf.matmul(h, w2)+b2)
#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = o))
step = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
#find accuracy
correct_prediction = tf.equal(tf.argmax(o,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(30000): #increase the number of iterations
    train_data = mnist.train.next_batch(128)
    _, t_loss = sess.run([step, cost], feed_dict={X:train_data[0], y:train_data[1]})
    if i%500 == 0:
        acc = sess.run([accuracy], feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print ("Step = {}, Accuracy = {}".format(i,acc))

