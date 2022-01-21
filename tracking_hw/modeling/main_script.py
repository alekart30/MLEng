import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
import shap
import matplotlib.pyplot as plt
import mlflow

mlflow.set_tracking_uri('http://tracking-server:5000')
experiment_id = mlflow.create_experiment("Random Forest Model")
with mlflow.start_run(experiment_id=experiment_id) as mlrun:
    seed = 30
    params_dict = {
        "n_estimators": 50,
        "max_depth": 1000,
        "class_weight": "balanced"
    }
    model = RandomForestClassifier(**params_dict, random_state=seed)
    
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
        df, df["default"], random_state=seed, stratify=df["default"]
    )
    
    mlflow.log_metric("Train size", X_train.shape[0])
    mlflow.log_metric("Test size", X_test.shape[0])

    # preprocessing
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # training
    model.fit(X_train_transformed, y_train)
    mlflow.log_params(params_dict)
    mlflow.set_tag("seed", seed)
    
    # evaluation
    pred_test = model.predict(X_test_transformed)
    mlflow.log_metric("f2-score", round(fbeta_score(y_test, pred_test, beta=2), 2))
    mlflow.sklearn.log_model(model, "classification_model")

    # feature importance
    short_names = [name[:30] for name in preprocessor.transformed_names_]
    rf_importance = pd.DataFrame({'importance' : model.feature_importances_,
                                  'name' : short_names})
    rf_importance.sort_values(by='importance', inplace=True)
    
    # save plot
    fig = plt.figure(figsize=(12,12))
    sns.barplot(data=rf_importance.tail(10), y="name", x="importance")
    plt.tight_layout()
    plt.savefig("feature_plot.png")
    mlflow.log_artifact("feature_plot.png")
