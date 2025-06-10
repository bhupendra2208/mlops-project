# src/train_model.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load sample dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Train a model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the trained model
joblib.dump(model, "model/model.pkl")
print("✅ Model saved successfully!")
