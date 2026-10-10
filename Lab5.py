# Lab 5: K-Nearest Neighbor (KNN) for Classification and Regression
# Dataset: Pima Indians Diabetes (diabetes.csv)
# Name: Anushka Boreklar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# =====================================================================
# PART A: KNN CLASSIFICATION (predict diabetes Outcome)
# =====================================================================

# 1. Load and explore the dataset
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.tail())
df.info()
print(df.describe())
print(df.isnull().sum())

# 2. Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 3. Zero values in columns where 0 is not physiologically valid
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = (df[zero_columns] == 0).sum()
print("Zero columns")
print(zero_counts)

# 4. Features and target
df_clean = df.copy()
X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]
print("X shape:", X.shape)
print("Y shape:", y.shape)

# 5. Train/test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training samples", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Training distribution")
print(y_train.value_counts(normalize=True))
print("Testing distribution")
print(y_test.value_counts(normalize=True))

# 6. Feature scaling (fit on training data only)
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# 7. Train KNN classifier (K = 5)
K = 5
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train_scaled, y_train)

# 8. Predict
y_pred = knn.predict(X_test_scaled)
print("Predictions:")
print(y_pred[:20])

# 9. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy:4f}")
precision = precision_score(y_test, y_pred)
print(f"Precision:{precision:4f}")
recall = recall_score(y_test, y_pred)
print(f"Recall:{recall:4f}")
f1 = f1_score(y_test, y_pred)
print(f"F1 score:{f1:4f}")

print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No diabetes", "Diabetes"],
            yticklabels=["No diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion matrix")
plt.show()

# 11. Effect of K on training and testing accuracy
k_values = range(1, 31)
train_accuracy = []
test_accuracy = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    train_accuracy.append(model.score(X_train_scaled, y_train))
    test_accuracy.append(model.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracy, marker="o", label="Training accuracy")
plt.plot(k_values, test_accuracy, marker="o", label="Testing accuracy")
plt.xlabel("Number of Neighbors(K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(k_values)
plt.legend()
plt.show()

best_k_acccuracy = k_values[np.argmax(test_accuracy)]
best_accuracy = max(test_accuracy)
print("Best K:", best_k_acccuracy)
print("Best Accuracy:", best_accuracy)

# 12. Cross-validation to choose K
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    cv_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), cv_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Mean Cross-validation Accuracy")
plt.title("KNN: K vs 10-fold Cross-validation Accuracy")
plt.xticks(range(1, 31))
plt.show()

best_k_cv = np.argmax(cv_scores) + 1
print("Best K based on cross validation:", best_k_cv)
print("Best cv Accuracy:", max(cv_scores))

# 13. Hyper-parameter tuning with GridSearchCV
param_grid = {
    "n_neighbors": list(range(3, 31, 2)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidian", "manhattan", "minkwoski"],
    "p": [1, 2]
}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid,
                           cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("Grid Search completed")
print("Best parameters:")
print(grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# =====================================================================
# PART B: KNN REGRESSION (predict BMI from the other clinical features)
# =====================================================================

# 1. Prepare data: remove rows with invalid BMI (0), median-impute other zero values
df_reg = df[df["BMI"] > 0].copy()
other_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]
df_reg[other_zero_cols] = df_reg[other_zero_cols].replace(0, np.nan)
df_reg[other_zero_cols] = df_reg[other_zero_cols].fillna(df_reg[other_zero_cols].median())

Xr = df_reg.drop(columns=["BMI", "Outcome"])
yr = df_reg["BMI"]
print("Regression X shape:", Xr.shape)
print("Regression y shape:", yr.shape)

# 2. Split and scale
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42)
scaler_r = StandardScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s = scaler_r.transform(Xr_test)

# 3. Train KNN regressor (K = 5) and predict
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(Xr_train_s, yr_train)
yr_pred = knn_reg.predict(Xr_test_s)

comparison = pd.DataFrame({"Actual BMI": yr_test.values[:10],
                           "Predicted BMI": yr_pred[:10].round(2)})
print(comparison)

# 4. Regression metrics
mae = mean_absolute_error(yr_test, yr_pred)
mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# 5. Effect of K on RMSE
rmse_list = []
for k in range(1, 31):
    m = KNeighborsRegressor(n_neighbors=k)
    m.fit(Xr_train_s, yr_train)
    rmse_list.append(np.sqrt(mean_squared_error(yr_test, m.predict(Xr_test_s))))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), rmse_list, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("RMSE")
plt.title("KNN Regression: K vs RMSE")
plt.xticks(range(1, 31))
plt.show()

best_k_reg = int(np.argmin(rmse_list)) + 1
print("Best K (lowest RMSE):", best_k_reg)
print("Lowest RMSE:", min(rmse_list))

# 6. Actual vs predicted plot using the best K
best_reg = KNeighborsRegressor(n_neighbors=best_k_reg)
best_reg.fit(Xr_train_s, yr_train)
yr_best_pred = best_reg.predict(Xr_test_s)
print("R2 with best K:", r2_score(yr_test, yr_best_pred))

plt.figure(figsize=(7, 6))
plt.scatter(yr_test, yr_best_pred, alpha=0.7)
lims = [yr_test.min(), yr_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("KNN Regression: Actual vs Predicted BMI")
plt.legend()
plt.show()
