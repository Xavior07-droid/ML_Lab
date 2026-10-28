# Lab 8: Support Vector Machine (SVM) classification
# Dataset: Mobile Price Classification (Kaggle) - train.csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("train.csv")
print(df.shape, df.isnull().sum().sum(), df.duplicated().sum())

X = df.drop("price_range", axis=1)
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), X.columns.tolist())])

linear_svm = Pipeline([("preprocessor", preprocessor),
                       ("classifier", SVC(kernel="linear"))])
linear_svm.fit(X_train, y_train)
print("Linear SVM Testing Accuracy:", accuracy_score(y_test, linear_svm.predict(X_test)))

rbf_svm = Pipeline([("preprocessor", preprocessor),
                    ("classifier", SVC(kernel="rbf"))])
rbf_svm.fit(X_train, y_train)
print("RBF SVM Testing Accuracy:", accuracy_score(y_test, rbf_svm.predict(X_test)))

pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", SVC())])
param_grid = [
    {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10]},
    {"classifier__kernel": ["rbf"], "classifier__C": [0.1, 1, 10],
     "classifier__gamma": ["scale", "auto"]},
]
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy",
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))
print("Training Accuracy =", best_model.score(X_train, y_train))
print("Testing Accuracy =", best_model.score(X_test, y_test))
