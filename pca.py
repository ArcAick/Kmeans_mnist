from keras.datasets import mnist
from keras.models import  Model
from keras.layers import Dense,Input
import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
from keras.utils import np_utils

nb_classes = 10

(x_test,y_test), (x_train,y_train) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')

def pca(data):
    #Transpose
    d = np.transpose(data)
    #Calcul puissance
    d_cov = np.cov(d)
    #idx et vect de chaque covariance
    i, v = np.linalg.eig(d_cov)
    #recuperere valeur reel
    i = np.real(i)
    v = np.real(v)

    #Recup valeur absolu
    tab = np.argsort(np.abs(i))[::-1]
    i = i[tab]
    v = v[tab]
    return i, v

if __name__ == '__main__':
     pca_i, pca_v = pca(x_train)
     vects = pca_v[:2]
     matriciel =  np.dot(x_train,np.transpose(vects))
     print(matriciel)
















