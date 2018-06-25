from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

from numpy import random
def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

# select parameters
hidden_size = [10, 25, 50, 60]
alpha = [0.05, 0.1, 0.25, 0.5]
lamb = [0.0001, 0.0005, 0.001, 0.01]

# setup the parameter selection function
def select_parameters(hidden_size, alpha, lamb, X, y):
    X_train, X_holdover, y_train, y_holdover = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_holdover, y_holdover, 
                                       test_size=0.5)
    # convert the targets (scalars) to vectors
    yv_train = convert_y_to_vect(y_train)
    yv_valid = convert_y_to_vect(y_valid)
    results = np.zeros((len(hidden_size)*len(alpha)*len(lamb), 4))
    cnt = 0
    for hs in hidden_size:
        for al in alpha:
            for l in lamb:
                nn_structure = [64, hs, 10]
                W, b, avg_cost = train_nn_batch(nn_structure, X_train, yv_train, 
                                    iter_num=3000, alpha=al, lamb=l)
                y_pred = predict_y(W, b, X_valid, 3)
                accuracy = accuracy_score(y_valid, y_pred) * 100
                print("Accuracy is {}% for {}, {}, {}".format(accuracy, hs, al, l))
                # store the data
                results[cnt, 0] = accuracy
                results[cnt, 1] = hs
                results[cnt, 2] = al
                results[cnt, 3] = l
                cnt += 1
    # get the index of the best accuracy
    best_idx = np.argmax(results[:, 0])
    return results, results[best_idx, :]

# def train_nn_batch(nn_structure, X, y, iter_num=3000, alpha=0.25):

def train_nn_batch_batch(nn_structure, X, y, iter_num=3000, alpha=0.25,lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer 
        for l in range(len(nn_structure) - 1, 0, -1):
            # W[l] += -alpha * (1.0/m * tri_W[l])
            # Now regularized
            W[l] += -alpha * (1.0/m * tri_W[l] + lamb * W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

def train_nn_SGD(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%50 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, 
            # to be used in the gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] = np.dot(delta[l+1][:,np.newaxis],
                                       np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] = delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (tri_W[l] + lamb * W[l])
                b[l] += -alpha * (tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

def train_nn_MBGD(nn_structure, X, y, bs=100, iter_num=3000, alpha=0.25, lamb=0.000):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        mini_batches = get_mini_batches(X, y, bs)
        for mb in mini_batches:
            X_mb = mb[0]
            y_mb = mb[1]
            # pdb.set_trace()
            for i in range(len(y_mb)):
                delta = {}
                # perform the feed forward pass and return the stored h and z values, 
                # to be used in the gradient descent step
                h, z = feed_forward(X_mb[i, :], W, b)
                # loop from nl-1 to 1 backpropagating the errors
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = calculate_out_layer_delta(y_mb[i,:], h[l], z[l])
                        avg_cost += np.linalg.norm((y_mb[i,:]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                        # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], 
                                          np.transpose(h[l][:,np.newaxis])) 
                        # trib^(l) = trib^(l) + delta^(l+1)
                        tri_b[l] += delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (1.0/bs * tri_W[l] + lamb * W[l])
                b[l] += -alpha * (1.0/bs * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target

## select parameters 
# print(select_parameters(hidden_size, alpha, lamb, X, y))


if __name__ == "__main__":
    # load data and scale
    digits = load_digits()
    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)
    y = digits.target
    X_train, X_holdover, y_train, y_holdover = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_holdover, y_holdover, test_size=0.5)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # convert digits to vectors
    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)
    # setup the NN structure
    nn_structure = [64, 30, 10]
    # train the NN
    W, b, avg_cost_func = train_nn_MBGD(nn_structure, X_train, y_v_train,iter_num=3000, alpha=0.25, lamb=0.000)
    # plot the avg_cost_func
    plt.plot(avg_cost_func)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()
    # get the prediction accuracy and print
    y_pred = predict_y(W, b, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
