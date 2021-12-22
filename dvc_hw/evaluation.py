from numpy import genfromtxt
import pickle
from sklearn.metrics import fbeta_score
import json

# read preprocessed data
X_test, y_test = genfromtxt("X_test_transformed.csv", delimiter=","), genfromtxt(
    "y_test.csv", delimiter=","
)

# load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# evaluate model
pred_test = model.predict(X_test)
metrics = {"f2-score": round(fbeta_score(y_test, pred_test, beta=2), 2)}

# save to file
with open("metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file)
