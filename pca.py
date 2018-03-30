from keras.datasets import mnist
from keras.models import  Model
from keras.layers import Dense,Input
import tensorflow as tf
from matplotlib import pyplot as plt
import sklearn.datasets

import numpy as np
from keras.utils import np_utils



def pca_for_mnist(data):
    mean_cols = np.mean(data.T, axis=1)
    center = data - mean_cols
    covariance = np.cov(center.T)
    vals, vects = np.linalg.eig(covariance)
    idx = np.argsort(vals)[::-1]
    vects = np.real(vects[idx])
    vals = np.real(vals[idx])
    project = vects.T.dot(center.T)
    return project

def pca(data):
    mean_cols = np.mean(data.T, axis=1)
    center = data - mean_cols
    covariance = np.cov(center.T)
    vals, vects = np.linalg.eig(covariance)
    project = vects.T.dot(center.T)
    return project


def run_pca_test():
    X1, Y1 = sklearn.datasets.make_blobs(n_samples=20000, n_features=2)
    test = np.array(X1)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    x = np.transpose(pca(test))
    ax.scatter(x[:,0],x[:,1], c=Y1)
    plt.show()

def run_pca_mnist():
    (x_test, y_test), (x_train, y_train) = mnist.load_data()
    n_rows = 60000

    x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')
    x_train  = x_train[:n_rows]
    mnist_data = pca_for_mnist(x_train)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    x = pca(mnist_data)
    ax.scatter(x[:, 0], x[:, 1], c=y_train[:n_rows])
    plt.show()


if __name__ == '__main__':
    #run_pca_test()
    run_pca_mnist()
















