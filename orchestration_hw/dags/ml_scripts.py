import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from numpy import genfromtxt
import pickle
import logging
import os

MODELS_PATH = "./models/"
DATA_PATH = "./data/"

def _define_last_model_version():
    models_list = os.listdir(MODELS_PATH)
    if len(models_list) == 0:
        return None
    else:
        names_without_ext = [os.path.splitext(model_name)[0] for model_name in models_list]
        last_version = max(model_name.split('_')[-1] for model_name in names_without_ext)
        return int(last_version[1:])

def _set_new_model_version():
    last_version = _define_last_model_version()
    if last_version is not None:
        return last_version + 1
    else:
        return 1
        
def train_model(**kwargs):
    # read preprocessed data
    X_train = pd.read_csv(DATA_PATH + 'X_train_transformed.csv')
    y_train = pd.read_csv(DATA_PATH + 'y_train.csv', index_col=0)

    logging.info("Input data successfully read")

    # fit model
    model = LogisticRegression(random_state=30, class_weight="balanced")
    model.fit(X_train, y_train)
    
    logging.info("Model successfully fitted")

    # save model
    try:
        version = kwargs['params']['model_version']
    except:
        version = _set_new_model_version()

    with open(MODELS_PATH + "model_v" + str(version) + ".pkl", "wb") as f:
        pickle.dump(model, f)
    
    logging.info("Model successfully saved")
    logging.info(f"Version: {version}")

def evaluate_on_train(**kwargs):
    # read preprocessed data
    X_train = pd.read_csv(DATA_PATH + 'X_train_transformed.csv')
    y_train = pd.read_csv(DATA_PATH + 'y_train.csv', index_col=0)

    logging.info("Input data successfully read")

    # load trained model
    try:
        version = kwargs['params']['model_version']
    except:
        version = _define_last_model_version()

    with open(MODELS_PATH + "model_v" + str(version) + ".pkl", "rb") as f:
        model = pickle.load(f)
    
    logging.info("Fitted model successfully loaded")
    logging.info(f"Version: {version}")

    # evaluate model
    pred_train = model.predict(X_train)

    logging.info(f"f2-score: {round(fbeta_score(y_train, pred_train, beta=2), 2)}")
  
def infer_predictions(**kwargs):
    # read preprocessed data
    X_test = pd.read_csv(DATA_PATH + 'X_test_transformed.csv')

    logging.info("Input data successfully read")

    # load trained model
    try:
        version = kwargs['params']['model_version']
    except:
        version = _define_last_model_version()

    with open(MODELS_PATH + "model_v" + str(version) + ".pkl", "rb") as f:
        model = pickle.load(f)
    
    logging.info("Fitted model successfully loaded")
    logging.info(f"Version: {version}")

    # evaluate model
    pred_test = model.predict(X_test)
    pd.DataFrame(pred_test).to_csv(DATA_PATH + "predictions.csv")

    logging.info("Inference is successfully done")

def evaluate_on_test():
    # read input data
    y_test = pd.read_csv(DATA_PATH + 'y_test.csv', index_col=0)
    y_pred = pd.read_csv(DATA_PATH + "predictions.csv", index_col=0)

    logging.info("Input data successfully read")

    # evaluate predictions
    logging.info(f"f2-score: {round(fbeta_score(y_test, y_pred, beta=2), 2)}")

