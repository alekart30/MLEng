from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt
import pickle

# read preprocessed data
X_train, y_train = genfromtxt('X_train_transformed.csv', delimiter=','), genfromtxt('y_train.csv', delimiter=',')

# fit model
model = LogisticRegression(random_state=30)
model.fit(X_train, y_train)

# save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
