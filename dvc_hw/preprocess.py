import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle

# load data
df = pd.read_csv("bank_scoring.csv")

# define transformations for features
dtypes = df.dtypes
numerical_features = dtypes[(dtypes == "int64") | (dtypes == "float64")].index.tolist()
categorical_features = dtypes[dtypes == "object"].index.tolist()
categorical_features.remove("foreing_worker")
numerical_features.remove('default')
binary_features = ["foreing_worker"]

numerical_def = gen_features(
    columns=[[c] for c in numerical_features],
    classes=[
        {'class': SimpleImputer, 'strategy': 'median'},
        {'class': MinMaxScaler}
    ]
)
categorical_def = gen_features(
    columns=[[c] for c in categorical_features],
    classes=[
        {'class': SimpleImputer, 'strategy': 'constant', "fill_value": "UNK"},
        {'class': OneHotEncoder, 'handle_unknown': 'ignore'}
    ]
)
binary_def = gen_features(
    columns=[[c] for c in binary_features],
    classes=[
        {'class': SimpleImputer, 'strategy': 'most_frequent'},
        {'class': OneHotEncoder, 'handle_unknown': 'error', 'drop': 'if_binary'}
    ]
)

features_def = numerical_def + categorical_def + binary_def
preprocessor = DataFrameMapper(features_def)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(df, df["default"], random_state=30, stratify=df["default"])

# preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# save preprocessed data for model training
np.savetxt("X_train_transformed.csv", X_train_transformed, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("X_test_transformed.csv", X_test_transformed, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")

# save preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
