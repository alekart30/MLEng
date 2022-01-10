import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
import mlflow

with mlflow.start_run() as mlrun:
    # load data
    df = pd.read_csv("bank_scoring.csv")
    # define transformations for features
    dtypes = df.dtypes
    numerical_features = dtypes[
        (dtypes == "int64") | (dtypes == "float64")
    ].index.tolist()
    categorical_features = dtypes[dtypes == "object"].index.tolist()
    categorical_features.remove("foreing_worker")
    numerical_features.remove("default")
    binary_features = ["foreing_worker"]

    numerical_def = gen_features(
        columns=[[c] for c in numerical_features],
        classes=[
            {"class": SimpleImputer, "strategy": "median"},
            {"class": MinMaxScaler},
        ],
    )
    categorical_def = gen_features(
        columns=[[c] for c in categorical_features],
        classes=[
            {"class": SimpleImputer, "strategy": "constant", "fill_value": "UNK"},
            {"class": OneHotEncoder, "handle_unknown": "ignore"},
        ],
    )
    binary_def = gen_features(
        columns=[[c] for c in binary_features],
        classes=[
            {"class": SimpleImputer, "strategy": "most_frequent"},
            {"class": OneHotEncoder, "handle_unknown": "error", "drop": "if_binary"},
        ],
    )

    features_def = numerical_def + categorical_def + binary_def
    preprocessor = DataFrameMapper(features_def)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, df["default"], random_state=30, stratify=df["default"]
    )
    
    mlflow.log_metric("Train size", X_train.shape[0])

    # preprocessing
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    class_weight = None
    mlflow.log_param("class_weight", class_weight)
    model = LogisticRegression(random_state=30, class_weight=class_weight)
    model.fit(X_train_transformed, y_train)

    pred_test = model.predict(X_test_transformed)
    mlflow.log_metric("f2-score", round(fbeta_score(y_test, pred_test, beta=2), 2))
    mlflow.sklearn.log_model(model, "classification_model")
