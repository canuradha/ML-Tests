import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

np.random.seed(42)

data = pd.read_csv('../datasets/iris/Iris.csv')
feature_set = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

features = data[feature_set]
clusters=3

# print(features.shape[1])

# # print(centeroids)

def init_centeroids(clusters, features):
    centeroids = np.zeros([clusters, features.shape[1]]).transpose()
    
    feature_max = features.loc[features[:].idxmax()].max()

    for idx,value in enumerate(feature_max):
        centeroids[idx] = np.random.randint(1, feature_max[idx], clusters)

    centeroids = centeroids.transpose()
    return centeroids


def compute_distance(features, clusters, centeroids):
    distance = np.zeros([features.shape[0], clusters])
    for i in range(clusters):
        distance[:, i] = norm(features - centeroids[i,:], axis=1)
    return distance

def within_cluter_sse(features, centeroids, labels):
    distances = np.zeros(features.shape[0])
    for i in range(clusters):
        distances[labels == i] = norm(features[i == labels] - centeroids[i], axis=1)
    return np.sum(np.absolute(distances))

def get_labels(distances):
    return np.argmin(distances, axis=1)

def update_centeroid(features, distances, clusters, labels):
    centeroids = np.zeros([clusters, features.shape[1]])
    for k in range(clusters):
        # print(np.mean(features[k == lables], axis=0))
        centeroids[k, :] = np.mean(features[labels == k], axis= 0)
    # print(within_cluter_sse(features,centeroids, labels))
    return centeroids

def centeroid_distance_change(old, new):
    return np.absolute(new - old)

def plot_data(features, centeroids, labels=None):
    # fig = plt.figure(figsize=(5,5))
    plt.clf()
    colormap = [np.random.rand(3,) for i in range(clusters)]
    if type(labels).__module__ != 'numpy':
        plt.scatter(features.iloc[:,0], features.iloc[:,1], color='k')
    else:
        for i, value in enumerate(np.array([features.iloc[:,0], features.iloc[:,1]]).transpose()):   
            plt.scatter(value[0], value[1], color=colormap[labels[i]])
    for i,point in enumerate(centeroids):
        plt.scatter(point[0], point[1], color=colormap[i], marker='*', s=200, edgecolor='k')
    plt.show()

def fit(features, clusters, sensivity = 0.001):
    centeroids = init_centeroids(clusters, features)
    plot_data(features, centeroids)
    i = 0
    while True:
        distances = compute_distance(features,clusters, centeroids)
        labels = get_labels(distances)
        centeroids_new = update_centeroid(features, distances, clusters, labels)
        # print(centeroid_distance_change(centeroids, centeroids_new))
        if((centeroid_distance_change(centeroids, centeroids_new)).any() <= sensivity):
            print('finished')
            break
        centeroids = centeroids_new
        if(i % 3 == 0):
            plot_data(features, centeroids, labels)
        i += 1
    plot_data(features, centeroids, labels)
    return centeroids



fit(features, clusters)
