import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


wine = pd.read_csv("../datasets/wine.data", header=None)

wine.columns = [ "Type", "Alcohol", "Malic acid", "Ash","Alcalinity of ash"  ,"Magnesium", "Total phenols",
 "Flavanoids","Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280_OD315" ,"Proline" ]

wine_features = wine.drop(["Type"], axis = 1)

wine_type = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(wine_features, wine_type, test_size = 0.2, random_state = 42)

reg_param = [ 1, 10, 50, 100, 500, 1000, 5000 ]
gamma = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

for g in gamma:
    model = svm.SVC(kernel='rbf', gamma=g)

    print ("training Model....")
    model.fit(X_train, y_train)

    predicted_types = model.predict(X_test)

    print("gamma = ", g, "Accuracy: ")
    print("accuracy: ", accuracy_score(y_test, predicted_types) )
    print("MSE: ", mean_squared_error(y_test, predicted_types))
