import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from collections import defaultdict

####### Dataset can be found at https://archive.ics.uci.edu/ml/datasets/Adult
adult = pd.read_csv("../datasets/US income/adult.data", header=None)


adult.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
 "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]
# names = pd.read_csv("lrn/datasets/US income/adult.names")
# adult = adult.dropna()
# adult = adult.sample(frac = 1).reset_index(drop = True)

# print(adult.head(10))

####### features to train 
features = adult.drop(['education', 'salary'], axis = 1)

####### target feature set encoding
target = adult['salary'].to_frame()
target = target.replace([' <=50K', ' >50K'], [0, 1])
target = target['salary']

# print(features.head(20))

####### determinig missing values
# for col in features.columns:
#     null_d = features.loc[features[col] == ' ?']
#     print(col, ':', null_d.shape)

######### Train test Split
fet_train,  fet_test, tar_train, tar_test = train_test_split(features, target, test_size= 0.2, random_state = 42)
# print(fet_train.head(10))

##### Categorical Encodings

########## lable encoding multiple columns
lbl_encoded_fet = ['workclass', 'sex']

encoder = defaultdict(LabelEncoder)

lbl_fet_train = fet_train[lbl_encoded_fet].apply(lambda x : encoder[x.name].fit_transform(x))
lbl_fet_test = fet_test[lbl_encoded_fet].apply(lambda x: encoder[x.name].transform(x))
# print(lbl_fet_test)

###### replacing original columns with concatenated columns
fet_train = pd.concat([fet_train.drop(lbl_encoded_fet, axis = 1), lbl_fet_train], axis = 1)
fet_test = pd.concat([fet_test.drop(lbl_encoded_fet, axis = 1), lbl_fet_test], axis = 1)


######  checking for an even distribution of selected columns for one hot encoding 
# print(features['relationship'].value_counts())          # somewhat okay
# print(features['race'].value_counts())                  # bad, not evenly distributed

##### One hot encoding 'relationships' using pandas
# print(fet_train['relationship'].unique())
dummy_train = pd.get_dummies(fet_train['relationship'], prefix = 'rel_')
dummy_test = pd.get_dummies(fet_test['relationship'], prefix = 'rel_')
# print(dummy_test.head(10))


###### filling cells if values that were missing in the training set but present in the test set with 0
dummy_test.reindex(columns = dummy_train.columns, fill_value = 0)

##### Replacing categorical column with encoded column
fet_train = fet_train.drop(['relationship'], axis = 1).join(dummy_train)
fet_test = fet_test.drop(['relationship'], axis = 1).join(dummy_test)

##### target encoding rest of the categorical features
target_enc = TargetEncoder()
fet_train = target_enc.fit_transform(fet_train, tar_train)
fet_test = target_enc.transform(fet_test)

# print(fet_train.head(10))

# print(fet_train.shape)

##### Training Random Forest and testing.

estimators = [10, 20, 50, 100, 250, 500, 1000]

# for est in estimators:
#     print("Estimator: ", est)
#     for leaves in estimators:

#         rf_classifier = RandomForestClassifier(random_state= 0, n_estimators=est, max_leaf_nodes = leaves)

#         rf_classifier.fit(fet_train, tar_train)

#         predictions = rf_classifier.predict(fet_test)

#         error = mean_absolute_error(predictions, tar_test)

#         print("Leaves : ",leaves, "  error: ",  error)

rf_classifier = RandomForestClassifier(random_state= 0, n_estimators=500, max_leaf_nodes = 500)
rf_classifier.fit(fet_train, tar_train)

############# Validating the model

adult_Validate = pd.read_csv("../datasets/US income/adult.test", header=None)
adult_Validate.columns = adult.columns
valid_feat = adult_Validate.drop(['education', 'salary'], axis = 1)
valid_target = adult_Validate['salary'].to_frame()
valid_target = valid_target.replace([' <=50K.', ' >50K.'], [0, 1])
valid_target = valid_target['salary']
# print(adult_Validate.head())

##### Encoding categorical data in the validation set

##Label Encoding
valid_label_enc =  valid_feat[lbl_encoded_fet].apply(lambda x: encoder[x.name].transform(x))
valid_feat = pd.concat([valid_feat.drop(lbl_encoded_fet, axis = 1), valid_label_enc], axis = 1)

## One hot encoding
dummy_valid = pd.get_dummies(valid_feat['relationship'], prefix = 'rel_')
dummy_valid.reindex(columns = dummy_train.columns, fill_value = 0)
valid_feat = valid_feat.drop(['relationship'], axis = 1).join(dummy_valid)

## Target Encoding
valid_feat = target_enc.transform(valid_feat)

# #####valid predictions

valid_predict = rf_classifier.predict(valid_feat)

valid_error = mean_absolute_error(valid_predict, valid_target)

print("error: ", valid_error)
