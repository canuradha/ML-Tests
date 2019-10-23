import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#### the dataset can be found at https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
data = pd.read_csv("../datasets/AB_NYC_2019.csv")

# s = (data.dtypes == 'object')
 
# print(list(s[s].index))

x_features = [
    'neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights', 
    'number_of_reviews', 'availability_365'
]

###### Selecting features without categorical variables.
x_data = data[x_features].select_dtypes(exclude=['object'])
y_data = data['price']

train_X, test_X, train_y, test_y = train_test_split(x_data, y_data, train_size=0.8, random_state=1)

ds_model = DecisionTreeRegressor(max_leaf_nodes=250 ,random_state=1)
ds_model.fit(train_X, train_y)

print("training complete \n\nTesting....")

val_predictions = ds_model.predict(test_X)

error = mean_absolute_error(val_predictions, test_y)

print("error: ", error)
print("Encoding and including Categorical Variables....")

###### Transforming categorical variables

label_x_data = data[x_features].copy()

###### Applying label encoder to "room_types" column
label_encoder = LabelEncoder()

label_x_data['room_type'] = label_encoder.fit_transform(label_x_data['room_type'])

###### Applying one hot encoder to "neighbourhood" column. since there's one column categories are given in a list

OH_encoder = OneHotEncoder(sparse=False, categories=[label_x_data['neighbourhood'].unique()])
# print(label_x_data['neighbourhood'].to_frame())

OH_cols = pd.DataFrame(OH_encoder.fit_transform(label_x_data['neighbourhood'].to_frame()))

###### Removing categorical data and rplacing it with encoded data 
num_x_data = label_x_data.drop(['neighbourhood'], axis=1)

x_data_encoded = pd.concat([num_x_data, OH_cols], axis = 1)

print("Encoding complete \n\nTrining Encoded Data...")

train_X, test_X, train_y, test_y = train_test_split(x_data_encoded, y_data, train_size=0.8, random_state=42)

###### the max_leaf_nodes are found as 50 by executing the commented code below
ds_model = DecisionTreeRegressor(max_leaf_nodes=50 ,random_state=42)

ds_model.fit(train_X, train_y)

encoded_predictions = ds_model.predict(test_X)

error = mean_absolute_error(encoded_predictions, test_y)

print("error: ", error)

####### Testing for optimal maximum leaf nodes in the tree
# candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# for tree_size in candidate_max_leaf_nodes:
#     ds_model = DecisionTreeRegressor(max_leaf_nodes=tree_size ,random_state=1)
#     ds_model.fit(train_X, train_y)

#     ds_model.fit(train_X, train_y)

#     encoded_predictions = ds_model.predict(test_X)

#     error = mean_absolute_error(encoded_predictions, test_y)

#     print("Tree Siez, error: ",tree_size, error)