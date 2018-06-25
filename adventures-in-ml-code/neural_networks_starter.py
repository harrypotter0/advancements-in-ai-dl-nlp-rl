import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)

# # Basic Sigmoid 
# f = 1 / (1 + np.exp(-x))
# plt.plot(x, f)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.show()

# # With weights
# w1 = 0.5 
# w2 = 1.0
# w3 = 2.0 
# l1 = 'w= 0.5'
# l2 = 'w = 1.0'
# l3 = 'w= 2.0'
# for w,l in [(w1,l1),(w2,l2),(w3,l3)]:
#     f = 1/(1+np.exp(-x*w))
#     plt.plot(x,f,label=l)
# plt.xlabel('x')
# plt.ylabel('h_w(x)')
# plt.legend(loc=2)
# plt.show()

# # Bias added
# w  =5.0
# b1 = -8.0
# b2 =0.0
# b3 = 8.0
# l1 = 'b=-8.0'
# l2 = 'b=0.0'
# l3 = 'b= 8.0'
# for b,l in [(b1,l1),(b2,l2),(b3,l3)]:
#     f = 1/(1+np.exp(-(x*w+b)))
#     plt.plot(x,f,label=l)
# plt.xlabel('x')
# plt.ylabel('h_wb(x)')
# plt.legend(loc=2)
# plt.show()

# feed-forward nn
import numpy as np
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])
def f(x):
    return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        h = np.zeros((w[l].shape[0],))
        for i in range(w[l].shape[0]):
            f_sum = 0
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            f_sum += b[l][i]
            h[i] = f(f_sum)
    return h

# w = [w1, w2]
# b = [b1, b2]
# #a dummy x input vector
# x = [1.5, 2.0, 3.0]

# h = simple_looped_nn_calc(3, x, w, b)
# print(h)

# For I-'python'
# %timeit simple_looped_nn_calc(3, x, w, b)

# Using Vectorized implementations
def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h

## Gradient Descent
x_old = 0 # The value does not matter as long as abs(x_new - x_old) > precision
x_new = 6 # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00001

# def df(x):
#     y = 4 * x**3 - 9 * x**2
#     return y

# while abs(x_new - x_old) > precision:
#     x_old = x_new
#     x_new += -gamma * df(x_old)

# print("The local minimum occurs at %f" % x_new)

# implementating NN in Python
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)
import matplotlib.pyplot as plt 
# plt.gray() 
# plt.matshow(digits.images[1]) 
# plt.show()
# Scale the data
# print(digits.data[0,:])
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
# print(X[0,:])

from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import numpy as np
print(y_test)
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
print(y_train[0], y_v_train[0])

# nn_structure = [64, 30, 10]
# def f(x):
#     return 1 / (1 + np.exp(-x))
# def f_deriv(x):
#     return f(x) * (1 - f(x))

# import numpy.random as r
# def setup_and_init_weights(nn_structure):
#     W = {}
#     b = {}
#     for l in range(1, len(nn_structure)):
#         W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
#         b[l] = r.random_sample((nn_structure[l],))
#     return W, b
# def init_tri_values(nn_structure):
#     tri_W = {}
#     tri_b = {}
#     for l in range(1, len(nn_structure)):
#         tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
#         tri_b[l] = np.zeros((nn_structure[l],))
#     return tri_W, tri_b
# def feed_forward(x, W, b):
#     h = {1: x}
#     z = {}
#     for l in range(1, len(W) + 1):
#         # if it is the first layer, then the input into the weights is x, otherwise, 
#         # it is the output from the last layer
#         if l == 1:
#             node_in = x
#         else:
#             node_in = h[l]
#         z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
#         h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
#     return h, z