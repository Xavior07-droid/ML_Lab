# Lab 7: Naive Bayes Classification using the Raisin Dataset
# Name: Yash Katekhaye | Section: A | Batch: A3 | Roll No: 39

# ---------------- Imports ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

pd.set_option('display.max_columns', None)

# ---------------- Load dataset ----------------
df = pd.read_excel('Raisin_Dataset.xlsx')
print(df.head())
print(df.tail())
df.info()
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
print("Duplicates:", df.duplicated().sum())
print(df['Class'].unique())
print(df['Class'].value_counts())

# ---------------- Bar plot ----------------
plt.figure(figsize=(10, 5))
sns.countplot(x="Class", data=df)
plt.title("Distribution of Raisin Classes")
plt.xlabel("Raisin Class")
plt.ylabel("Number of Samples")
plt.show()

# ---------------- Pie chart ----------------
class_counts = df['Class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Raisin Classes')
plt.show()

numerical_features = df.select_dtypes(include=np.number).columns
print(numerical_features.tolist())

# ---------------- Histograms ----------------
df[numerical_features].hist(figsize=(15, 10))
plt.suptitle("Distribution of Numerical Features", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# ---------------- Box plots ----------------
plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.show()

# ---------------- Histogram with KDE by class ----------------
for column in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, hue="Class", kde=True, bins=25)
    plt.title(f"{column} Distribution by Class")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# ---------------- Correlation matrix ----------------
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ---------------- Scatter plots ----------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Area', y='Perimeter', hue='Class',
                palette={'Kecimen': 'green', 'Besni': 'blue'})
plt.title("Area vs Perimeter")
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MajorAxisLength', y='MinorAxisLength', s=70, hue='Class')
plt.title("MajorAxisLength vs MinorAxisLength")
plt.show()

# ---------------- Data preprocessing ----------------
label_encoder = LabelEncoder()
df['Class_encoded'] = label_encoder.fit_transform(df['Class'])
print(df[['Class', 'Class_encoded']].head(10))

print("Class Mapping:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index} : {class_name}")

X = df[numerical_features]
y = df["Class_encoded"]
print("X shape", X.shape)
print("Y shape", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ---------------- Train Gaussian Naive Bayes ----------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ---------------- Prediction ----------------
y_pred = gnb.predict(X_test)
print(y_pred)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))
comparison["Actual Class"] = label_encoder.inverse_transform(comparison["Actual"])
comparison["Predicted Class"] = label_encoder.inverse_transform(comparison["Predicted"])
print(comparison.head(10))

# ---------------- Evaluation ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=========MODEL PERFORMANCE=========")
print("Accuracy:", accuracy * 100, "%")
print("Precision:", precision * 100, "%")
print("Recall:", recall * 100, "%")
print("F1 Score:", f1 * 100, "%")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.show()

# ---------------- Training vs testing accuracy ----------------
y_train_pred = gnb.predict(X_train)
y_test_pred = gnb.predict(X_test)
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred) * 100, "%")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred) * 100, "%")

# ---------------- Hyperparameter: var_smoothing ----------------
print(gnb.get_params())
smoothing_values = [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
result = []
for var in smoothing_values:
    model = GaussianNB(var_smoothing=var)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    result.append([var,
                   accuracy_score(y_test, prediction),
                   precision_score(y_test, prediction),
                   recall_score(y_test, prediction),
                   f1_score(y_test, prediction)])

results_df = pd.DataFrame(result, columns=["var_smoothing", "accuracy", "precision", "recall", "f1"])
print(results_df)

# ---------------- Predict a new sample (realistic values) ----------------
new_sample = pd.DataFrame({
    "Area": [87524],
    "MajorAxisLength": [442.246011],
    "MinorAxisLength": [253.291155],
    "Eccentricity": [0.819738],
    "ConvexArea": [90546],
    "Extent": [0.758651],
    "Perimeter": [1184.040],
})
prediction = gnb.predict(new_sample)
print("Predicted Class:", label_encoder.inverse_transform(prediction)[0])
