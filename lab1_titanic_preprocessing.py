# Lab 1: Data Preprocessing on Titanic Dataset
# Data Preprocessing using Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("Titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nShape of dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nStatistical description:")
print(df.describe())

# ---------------------------------------------------------
# 2. Check Missing Values
# ---------------------------------------------------------
print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values")
plt.show()

# ---------------------------------------------------------
# 3. Handle Missing Values
# ---------------------------------------------------------
# Age -> median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked -> mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has a large number of missing values in the Titanic dataset.
# Drop it because it is not used in the final feature set.
df = df.drop(columns=["Cabin"])

print("\nMissing values after handling:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 4. Check Duplicate Rows
# ---------------------------------------------------------
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# ---------------------------------------------------------
# 5. Detect Outliers using Boxplots
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.show()

# ---------------------------------------------------------
# 6. Remove Fare Outliers using IQR
# ---------------------------------------------------------
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot After IQR Outlier Removal")
plt.show()

# ---------------------------------------------------------
# 7. Remove Age Outliers using IQR
# ---------------------------------------------------------
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot After IQR Outlier Removal")
plt.show()

# ---------------------------------------------------------
# 8. Categorical Encoding
# ---------------------------------------------------------
# Encode Sex: male = 0, female = 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

print("\nData after encoding:")
print(df.head())

# ---------------------------------------------------------
# 9. Univariate Analysis
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df["Fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", data=df)
plt.title("Pclass Count")
plt.show()

# Embarked columns after one-hot encoding
embarked_cols = [c for c in ["Embarked_S", "Embarked_Q", "Embarked_C"] if c in df.columns]

if embarked_cols:
    embarked_counts = df[embarked_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=embarked_counts.index, y=embarked_counts.values)
    plt.title("Embarked Count")
    plt.xlabel("Embarked")
    plt.ylabel("Count")
    plt.show()

# ---------------------------------------------------------
# 10. Bivariate Analysis
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survived")
plt.show()

if "Embarked_S" in df.columns:
    temp_embarked = pd.DataFrame({
        "Embarked": pd.Series(
            np.select(
                [
                    df["Embarked_S"].eq(1),
                    df.get("Embarked_Q", pd.Series(0, index=df.index)).eq(1),
                    df.get("Embarked_C", pd.Series(0, index=df.index)).eq(1)
                ],
                ["S", "Q", "C"],
                default="Unknown"
            ),
            index=df.index
        ),
        "Survived": df["Survived"]
    })

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Embarked", hue="Survived", data=temp_embarked)
    plt.title("Embarked vs Survived")
    plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Pclass vs Survived")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="Age", y="Fare", data=df)
plt.title("Age vs Fare")
plt.show()

# ---------------------------------------------------------
# 11. Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="inferno"
)

plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------
# 12. Feature Scaling
# ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Drop identifier/text columns that are not used as numerical ML features.
X = X.drop(columns=["PassengerId", "Name", "Ticket"], errors="ignore")

# Convert any remaining boolean columns to integers.
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(int)

# Standardize numerical features.
scale_cols = [col for col in ["Age", "SibSp", "Parch", "Fare"] if col in X.columns]

scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

print("\nScaled feature data:")
print(X.head(15))

# ---------------------------------------------------------
# 13. Train-Test Split
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nFirst rows of X_train:")
print(X_train.head())
