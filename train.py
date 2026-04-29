import numpy as np
import pandas as pd
import pickle
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = sklearn.datasets.load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['label'] = data.target

# Split
X = df.drop('label', axis=1)
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, Y_train)

# Accuracy
print("Train Accuracy:", accuracy_score(Y_train, pipeline.predict(X_train)))
print("Test Accuracy:", accuracy_score(Y_test, pipeline.predict(X_test)))

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved!")
