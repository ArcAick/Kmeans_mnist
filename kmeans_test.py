import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from copy import deepcopy

def distance(a,b):
    np.sqrt(sum((a-b)**2))

def main():
    data = pd.read_csv('./test.csv')
    f1 = data['feat1']
    f2 = data['feat2']
    dataset = np.array(list(zip(f1, f2)))
    k = 2

    # plt.scatter(f1,f2)
    # plt.show()

    #Random Centroid
    x = np.random.randint(0,np.max(dataset),size=k)
    y = np.random.randint(0,np.max(dataset),size=k)
    centroids = np.array(list(zip(x, y)), dtype=np.float32)

    #Init Centroid and Cluster
    c = np.zeros(centroids.shape)
    classes = np.zeros(len(dataset))

    print("Calcul distance")

    #Distance
    dist= distance(centroids,c)

    print("Calcul centroid for each points")

    while dist!= 0:
        for i in range(len(dataset)):
            distances= distance(dataset[i], centroids)
            cluster = np.min(distances)
            classes[i]= cluster
        c_list = [c for c in centroids]

        print("Calcul centroid by cluster")

        for i in range(k):
            p = [dataset[j] for j in range(len(dataset)) if classes[j]==i]
            centroids[i]= np.mean(p)
        distance(centroids,c_list)

    fig, a = plt.subplots()
    for i in range(k):
        points = np.array([dataset[j] for j in range(len(dataset)) if classes[j] == i])
        a.scatter(points[:, 0], points[:, 1])
    a.scatter(centroids[:, 0], centroids[:, 1])
    a.show()

if __name__ == '__main__':
    main()