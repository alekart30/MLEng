import sys
sys.path.append('..')
from preprocess import load_data
from numpy import genfromtxt
from sklearn.metrics import fbeta_score
import pickle

class TestClass:
    def test_data_loading(self):
        df = load_data("../bank_scoring.csv")
        assert (df.shape[0] > 0) and (df.shape[1]) > 0

    def test_columns_number(self):
        df = load_data("../bank_scoring.csv")
        assert df.shape[1] == 21

    def test_preprocessed_data(self):
        X_train = genfromtxt('../X_train_transformed.csv', delimiter=',')
        X_test = genfromtxt('../X_test_transformed.csv', delimiter=',')
        assert (X_train.shape[1] == 60) and (X_test.shape[1] == 60) 
    
    def test_model_able_to_predict(self):
        with open('../model.pkl', 'rb') as f:
            model = pickle.load(f)
        X_test = genfromtxt('../X_test_transformed.csv', delimiter=',')
        model.predict(X_test)
        assert True

    def test_model_performance_threshold(self):
        with open('../model.pkl', 'rb') as f:
            model = pickle.load(f)
        X_test, y_test = genfromtxt('../X_test_transformed.csv', delimiter=','), genfromtxt('../y_test.csv', delimiter=',')
        pred_test = model.predict(X_test)
        assert fbeta_score(y_test, pred_test, beta=2) > 0.5