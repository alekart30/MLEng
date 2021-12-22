import shap
import pickle
from numpy import genfromtxt
import matplotlib.pyplot as plt

# read preprocessed data
X_train, X_test = genfromtxt("X_train_transformed.csv", delimiter=","),
                  genfromtxt("X_test_transformed.csv", delimiter=",")

# load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

short_names = [name[:10] for name in preprocessor.transformed_names_]
# explain predictions
explainer = shap.Explainer(model, X_train, feature_names=short_names)
shap_values = explainer(X_test)

# save plot
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("feature_plot.png")
