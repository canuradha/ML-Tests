import pandas as pd
import numpy as np
from numpy.linalg import norm

np.random.seed(42)

data = pd.read_csv('../Data/Iris.csv')
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

def update_centeroid(features, distances, clusters):
    centeroids = np.zeros([clusters, features.shape[1]])
    lables = np.argmin(distances, axis=1)
    # print(lables)
    for k in range(clusters):
        # print(np.mean(features[k == lables], axis=0))
        centeroids[k, :] = np.mean(features[lables == k], axis= 0)
    print(within_cluter_sse(features,centeroids, lables))
    return centeroids

def centeroid_distance_change(old, new):
    return np.absolute(old - new)

def fit(features, clusters, sensivity = 0.001):
    centeroids = init_centeroids(clusters, features)
    while True:
        distances = compute_distance(features,clusters, centeroids)
        centeroids_new = update_centeroid(features, distances, clusters)
        print(centeroid_distance_change(centeroids, centeroids_new))
        if((centeroid_distance_change(centeroids, centeroids_new)).any() <= sensivity):
            break
        centeroids = centeroids_new
    return centeroids



fit(features, clusters)