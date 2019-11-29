import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


wine = pd.read_csv("../datasets/wine.data", header=None)

wine.columns = [ "Type", "Alcohol", "Malic acid", "Ash","Alcalinity of ash"  ,"Magnesium", "Total phenols",
 "Flavanoids","Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280_OD315" ,"Proline" ]

wine_features = wine.drop(["Type"], axis = 1)

wine_type = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(wine_features, wine_type, test_size = 0.3, random_state = 42)

reg_param = [ 1, 10, 50, 100, 500, 1000, 5000 ]
gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

# for g in gamma:
#     model = svm.SVC(kernel='linear', gamma=g)

#     print ("training Model....")
#     model.fit(X_train, y_train)

#     predicted_types = model.predict(X_test)

#     print("gamma = ", g, "Accuracy: ")
#     print("accuracy: ", accuracy_score(y_test, predicted_types) )
#     print("MSE: ", mean_squared_error(y_test, predicted_types))

model = svm.SVC(kernel='linear', gamma=0.01)
model.fit(X_train, y_train)

predicts = model.predict(X_test)

# --------------- Visualizing the predctions--------------------

fig, (ax1, ax2) = plt.subplots(1,2)    
X_test.plot.scatter(x='Alcohol', y='OD280_OD315', c=predicts, colormap='Dark2', ax= ax1, title = 'Predicts' )

# plt.subplot(1,2,2)
X_test.plot.scatter(x='Alcohol', y='OD280_OD315', c=y_test, colormap='Dark2' , ax= ax2, title= 'Real Data')

plt.show()
# plt.scatter(features['Alcohol'], X_test['Color intensity'], color=colormap[predicts])