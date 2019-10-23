import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#### the dataset can be found at https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
data = pd.read_csv("../datasets/AB_NYC_2019.csv")

# print(data['neighbourhood'].value_counts())
# print(data['neighbourhood'].value_counts().plot())
# plt.show()

x_features = [
    'neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights', 
    'number_of_reviews', 'availability_365'
]

x_data = data[x_features]
y_data = data['price']

##### Applying label encoder to "room_types" column
label_encoder = LabelEncoder()

x_data['room_type'] = label_encoder.fit_transform(x_data['room_type'])

x_data_w_o_cat = x_data.drop(['neighbourhood'], axis=1)

##### Applying one hot encoder to "neighbourhood" column. since there's one column categories are given in a list
OH_encoder = OneHotEncoder(sparse=False, categories=[x_data['neighbourhood'].unique()])
# print(label_x_data['neighbourhood'].to_frame())

OH_cols = pd.DataFrame(OH_encoder.fit_transform(x_data['neighbourhood'].to_frame()))

###### Removing and adding categorical data with encoded data in the original data
num_x_data = x_data.drop(['neighbourhood'], axis=1)

x_data_encoded = pd.concat([num_x_data, OH_cols], axis = 1)

print("Encoding complete \n\nTrining Encoded Data...")

############ Training model with encoded categorical varibles replaced
# train_X, test_X, train_y, test_y = train_test_split(x_data_encoded, y_data, train_size=0.8, random_state=42)

############ Training model without categorical variables (without 'neighbourhood')
train_X, test_X, train_y, test_y = train_test_split(x_data_w_o_cat, y_data, train_size=0.8, random_state=42)

###### training the random forest & Determining the optimum no of extimators (descion trees)

estimators = [10, 20, 50, 100, 250, 500, 1000]

for est in estimators:

    rf_regressor = RandomForestRegressor(random_state= 0, n_estimators=est)

    rf_regressor.fit(train_X, train_y)

    predictions = rf_regressor.predict(test_X)

    error = mean_absolute_error(predictions, test_y)

    print("Estimators, error: ", est, error)


##### Vizualizing the results.

# X_grid = np.arange(min(test_X['minimum_nights']), max(test_X['minimum_nights']), 1)
# # print(X_grid)
# X_grid = X_grid.reshape((len(X_grid), 1))



# plt.scatter(test_X['minimum_nights'], test_y, color='red')
# plt.scatter(test_X['minimum_nights'], predictions, color='blue')
# plt.show()