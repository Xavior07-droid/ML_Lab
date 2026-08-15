# ML Lab 2: Regression on USA Housing Dataset
# Simple Linear Regression, Multiple Linear Regression, Ridge Regression and Lasso Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load and inspect the dataset
# ---------------------------------------------------------
df = pd.read_csv("USA Housing.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Address is an identifier/text field and is not used for regression.
df.drop("Address", axis=1, inplace=True)

# Optional display formatting used in the lab.
pd.set_option("display.float_format", lambda x: f"{x:.2f}")
print("\nDataset after removing Address:")
print(df.head())

# ---------------------------------------------------------
# 2. Correlation heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Boxplots for Price and Avg. Area Income
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x=df["Price"])
plt.title("Price Boxplot")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x=df["Avg. Area Income"])
plt.title("Avg. Area Income Boxplot")
plt.show()

# ---------------------------------------------------------
# 4. Remove Price outliers using IQR
# ---------------------------------------------------------
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Price"] >= lower) & (df["Price"] <= upper)].copy()

# ---------------------------------------------------------
# 5. Remove Avg. Area Income outliers using IQR
# ---------------------------------------------------------
Q1 = df["Avg. Area Income"].quantile(0.25)
Q3 = df["Avg. Area Income"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Avg. Area Income"] >= lower)
    & (df["Avg. Area Income"] <= upper)
].copy()

# ---------------------------------------------------------
# 6. Distribution of Avg. Area Income
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["Avg. Area Income"], bins=30, edgecolor="black")
plt.title("Distribution of Avg. Area Income")
plt.xlabel("Avg. Area Income")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------------
# 7. Simple Linear Regression
#    Feature: Avg. Area Income
#    Target: Price
# ---------------------------------------------------------
X = df[["Avg. Area Income"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

slr_model = LinearRegression()
slr_model.fit(X_train, y_train)

y_pred = slr_model.predict(X_test)

print("\nSimple Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
slr_r2 = r2_score(y_test, y_pred)
print("R2 Score:", slr_r2)
print("Intercept:", slr_model.intercept_)
print("Slope:", slr_model.coef_[0])

# Actual data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, linewidth=2, label="Regression Line")
plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# ---------------------------------------------------------
# 8. Simple Linear Regression prediction for new input
# ---------------------------------------------------------
# Example: enter the Average Area Income in the same units as the dataset.
try:
    income = float(input("Enter Average Area Income: "))
    new_data = np.array([[income]])
    prediction = slr_model.predict(new_data)
    print(f"Predicted House Price = ${prediction[0]:,.2f}")
except ValueError:
    print("Please enter a numeric value for Average Area Income.")

# ---------------------------------------------------------
# 9. Multiple Linear Regression
# ---------------------------------------------------------
features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]

X = df[features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

predictions = mlr_model.predict(X_test)

print("\nMultiple Linear Regression Results")
print("Coefficients:")
print(pd.DataFrame({"Feature": features, "Coefficient": mlr_model.coef_}))
print("Intercept:", mlr_model.intercept_)
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2 Score:", r2_score(y_test, predictions))

# Actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, label="Predictions")
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2,
    label="Perfect Fit"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# ---------------------------------------------------------
# 10. Multiple Linear Regression prediction for new input
# ---------------------------------------------------------
try:
    income = float(input("Enter Average Area Income: "))
    house_age = float(input("Enter Average Area House Age: "))
    rooms = float(input("Enter Average Area Number of Rooms: "))
    bedrooms = float(input("Enter Average Number of Bedrooms: "))
    population = float(input("Enter Area Population: "))

    input_array = np.array([[income, house_age, rooms, bedrooms, population]])
    predicted_price = mlr_model.predict(input_array)
    print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
except ValueError:
    print("Please enter numeric values for all features.")

# ---------------------------------------------------------
# 11. Ridge Regression - default model
# ---------------------------------------------------------
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nDefault Ridge Regression R2:", r2_score(y_test, ridge_pred))

# ---------------------------------------------------------
# 12. Ridge Regression with GridSearchCV
# ---------------------------------------------------------
param_grid_ridge = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]
}

grid_ridge = GridSearchCV(
    estimator=Ridge(),
    param_grid=param_grid_ridge,
    scoring="r2",
    cv=5
)
grid_ridge.fit(X_train, y_train)

print("Best Ridge Parameters:", grid_ridge.best_params_)
print("Best Ridge CV Score:", grid_ridge.best_score_)

ridge_best_pred = grid_ridge.predict(X_test)
print("Ridge Test R2:", r2_score(y_test, ridge_best_pred))

# ---------------------------------------------------------
# 13. Lasso Regression with GridSearchCV
# ---------------------------------------------------------
param_grid_lasso = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]
}

grid_lasso = GridSearchCV(
    estimator=Lasso(max_iter=5000),
    param_grid=param_grid_lasso,
    scoring="r2",
    cv=5
)
grid_lasso.fit(X_train, y_train)

print("\nBest Lasso Parameters:", grid_lasso.best_params_)
print("Best Lasso CV Score:", grid_lasso.best_score_)

lasso_pred = grid_lasso.predict(X_test)
print("Lasso Test R2:", r2_score(y_test, lasso_pred))

# ---------------------------------------------------------
# 14. Final model comparison
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Simple Linear Regression", "Multiple Linear Regression", "Ridge", "Lasso"],
    "R2 Score": [
        slr_r2,
        r2_score(y_test, predictions),
        r2_score(y_test, ridge_best_pred),
        r2_score(y_test, lasso_pred),
    ]
})

print("\nModel comparison:")
print(comparison.to_string(index=False))
