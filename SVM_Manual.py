import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('Iris.csv')

x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x, setosa_y, marker='+', color='blue')
plt.scatter(versicolor_x, versicolor_y, marker='_', color='green')
# plt.show()

Y = []

df = df.sample(frac=1).reset_index(drop=True)
# print(df)

target = df['Species']
for sp in target:
    if(sp == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)

X = df.loc[:,'SepalLengthCm':'PetalLengthCm'].values.tolist()
# print(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train).reshape(80,1)
y_test = np.array(y_test).reshape(20,1)

## SVM 

train_feature_1 = x_train[:,0].reshape(80,1)
train_feature_2 = x_train[:,1].reshape(80,1)

w1 = np.zeros((80,1))
w2 = np.zeros((80,1))

epochs = 1
alpha = 0.0001
prod = []
while(epochs < 1000):
    y = w1* train_feature_1 + w2 * train_feature_2
    prod = y* y_train
    # print(epochs)
    count = 0
    for val in prod:
        if(val >= 1):
            cost = 0
            w1 = w1 - alpha * (2 * 1/epochs * w1)
            w2 = w2 - alpha * (2 * 1/epochs * w2)
        else:
            cost = 1 - val 
            w1 = w1 + alpha * (train_feature_1[count] * y_train[count] - 2 * 1/epochs * w1)
            w2 = w2 + alpha * (train_feature_2[count] * y_train[count] - 2 * 1/epochs * w2)
        count += 1
    epochs += 1

# test the data

from sklearn.metrics import accuracy_score

#clip the trained data to fit the test data
index = list(range(20,80))
w1 = np.delete(w1,index)
w2 = np.delete(w2,index)

w1 = w1.reshape(20,1)
w2 = w2.reshape(20,1)
## Extract the test data features 
test_f1 = x_test[:,0].astype(int).reshape(20,1)
test_f2 = x_test[:,1].astype(int).reshape(20,1)

## Predict
y_pred = w1 * test_f1 + w2 * test_f2
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(y_test,predictions))