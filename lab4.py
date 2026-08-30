# ============================================================
# ML LAB 4 - WINE QUALITY PREDICTION
# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    learning_curve
)

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# Keep winequality-red.csv in the same folder as this file

df = pd.read_csv("winequality-red.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())


# ------------------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. CHECK DUPLICATE VALUES
# ------------------------------------------------------------

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# 6. DISPLAY COLUMN NAMES
# ------------------------------------------------------------

print("\n========== COLUMN NAMES ==========")
print(df.columns)


# ------------------------------------------------------------
# 7. UNIVARIATE ANALYSIS - HISTOGRAM
# ------------------------------------------------------------

numeric_cols = df.columns

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. UNIVARIATE ANALYSIS - BOXPLOT
# ------------------------------------------------------------

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

features = df.drop(columns=["quality"])

outlier_summary = []

for col in features.columns:

    # Calculate Q1 and Q3
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = (
        (features[col] < lower_limit) |
        (features[col] > upper_limit)
    )

    # Count outliers
    outlier_count = outliers.sum()

    # Percentage of outliers
    outlier_percentage = (
        outlier_count / len(features)
    ) * 100

    # Store results
    outlier_summary.append([
        col,
        Q1,
        Q3,
        IQR,
        lower_limit,
        upper_limit,
        outlier_count,
        outlier_percentage
    ])


# Create outlier table

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=[
        "Feature",
        "Q1",
        "Q3",
        "IQR",
        "Lower Limit",
        "Upper Limit",
        "Outlier Count",
        "Outlier Percentage"
    ]
)

print("\n========== OUTLIER SUMMARY ==========")
print(outlier_df)


# ------------------------------------------------------------
# 10. PRINT OUTLIERS FOR EACH FEATURE
# ------------------------------------------------------------

print("\n========== OUTLIERS IN EACH FEATURE ==========")

for column, count in zip(
    outlier_df["Feature"],
    outlier_df["Outlier Count"]
):
    print(f"{column:25s}: {count}")


# ------------------------------------------------------------
# 11. OUTLIER COUNT VISUALIZATION
# ------------------------------------------------------------

outlier_count = outlier_df.set_index("Feature")["Outlier Count"]

plt.figure(figsize=(12, 6))

outlier_count.sort_values(
    ascending=False
).plot(kind="bar")

plt.title("Number of Outliers in Each Feature")
plt.xlabel("Features")
plt.ylabel("Number of Outliers")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. DISTRIBUTION OF WINE QUALITY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(x=df["quality"])

plt.title("Distribution of Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. QUALITY VALUE COUNTS
# ------------------------------------------------------------

print("\n========== QUALITY VALUE COUNTS ==========")
print(df["quality"].value_counts().sort_index())


# ------------------------------------------------------------
# 14. BIVARIATE ANALYSIS
# ------------------------------------------------------------

# Alcohol vs Quality

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="quality",
    y="alcohol",
    data=df
)

plt.title("Alcohol vs Quality")
plt.xlabel("Wine Quality")
plt.ylabel("Alcohol")
plt.tight_layout()
plt.show()


# Fixed Acidity vs Quality

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="quality",
    y="fixed acidity",
    data=df
)

plt.title("Fixed Acidity vs Wine Quality")
plt.xlabel("Wine Quality")
plt.ylabel("Fixed Acidity")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Volatile Acidity vs Quality

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="volatile acidity",
    y="quality",
    data=df
)

plt.title("Volatile Acidity vs Wine Quality")
plt.xlabel("Volatile Acidity")
plt.ylabel("Wine Quality")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. MULTIVARIATE ANALYSIS - CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16. PAIRPLOT
# ------------------------------------------------------------

selected = [
    "alcohol",
    "volatile acidity",
    "sulphates",
    "citric acid",
    "quality"
]

sns.pairplot(
    df[selected],
    hue="quality"
)

plt.show()


# ------------------------------------------------------------
# 17. ALCOHOL DISTRIBUTION ACROSS QUALITY GROUPS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="quality",
    y="alcohol",
    data=df
)

plt.title("Alcohol Distribution Across Quality Groups")
plt.xlabel("Quality Group")
plt.ylabel("Alcohol")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 18. CREATE QUALITY GROUP
# ------------------------------------------------------------

df["quality_group"] = pd.cut(
    df["quality"],
    bins=[0, 5, 6, 10],
    labels=["Low", "Medium", "High"]
)

print("\n========== QUALITY GROUP ==========")
print(df[["quality", "quality_group"]].head())


# ------------------------------------------------------------
# 19. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(
    columns=["quality", "quality_group"],
    errors="ignore"
)

y = df["quality"]

print("\n========== FEATURES ==========")
print(X.columns)

print("\n========== FEATURE SHAPE ==========")
print(X.shape)

print("\n========== TARGET SHAPE ==========")
print(y.shape)

print("\n========== TARGET DISTRIBUTION ==========")
print(y.value_counts().sort_index())


# ------------------------------------------------------------
# 20. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ------------------------------------------------------------
# 21. DECISION TREE CLASSIFIER
# ------------------------------------------------------------

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt_model.fit(X_train, y_train)


# ------------------------------------------------------------
# 22. PREDICTION
# ------------------------------------------------------------

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)


# ------------------------------------------------------------
# 23. MODEL EVALUATION
# ------------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

print("\n========== DECISION TREE RESULTS ==========")

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy :", test_accuracy)
print("Precision         :", precision)
print("Recall            :", recall)
print("F1 Score          :", f1)


# ------------------------------------------------------------
# 24. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 25. CONFUSION MATRIX
# ------------------------------------------------------------

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 26. DECISION TREE DEPTH AND LEAVES
# ------------------------------------------------------------

print("\n========== TREE INFORMATION ==========")

print(
    "Tree depth:",
    dt_model.get_depth()
)

print(
    "Number of leaves:",
    dt_model.get_n_leaves()
)


# ------------------------------------------------------------
# 27. VISUALIZE DECISION TREE
# ------------------------------------------------------------

plt.figure(figsize=(25, 15))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=[
        str(c)
        for c in sorted(y.unique())
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Entropy")
plt.show()


# ------------------------------------------------------------
# 28. DECISION TREE WITH MAX DEPTH = 3
# ------------------------------------------------------------

dt_depth3 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

dt_depth3.fit(X, y)

plt.figure(figsize=(25, 15))

plot_tree(
    dt_depth3,
    feature_names=X.columns,
    class_names=[
        str(c)
        for c in sorted(y.unique())
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree - Entropy (Max Depth = 3)"
)

plt.show()


# ------------------------------------------------------------
# 29. ROC-AUC CURVE BEFORE HYPERPARAMETER TUNING
# ------------------------------------------------------------

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predict probabilities

y_prob = dt_model.predict_proba(X_test)

# Get classes

classes = dt_model.classes_

# Binarize test labels

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

# Overall multiclass AUC

auc_score = roc_auc_score(
    y_test_bin,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print(
    "\nAUC before hyperparameter tuning:",
    auc_score
)


# Plot ROC curve

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Class {class_name} "
              f"(AUC = {class_auc:.3f})"
    )


# Random classifier

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Before Hyperparameter Tuning")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# 30. HYPERPARAMETER TUNING
# ------------------------------------------------------------

dt = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)


# Parameters to test

param_grid = {
    "criterion": ["entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"],
    "ccp_alpha": [0, 0.001, 0.005, 0.01]
}


# Grid Search

grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc_ovr",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 31. BEST PARAMETERS
# ------------------------------------------------------------

print("\n========== BEST PARAMETERS ==========")

print(
    grid_search.best_params_
)

print("\nBest Cross-Validation AUC:")

print(
    grid_search.best_score_
)


# ------------------------------------------------------------
# 32. BEST TREE
# ------------------------------------------------------------

best_tree = grid_search.best_estimator_


# ------------------------------------------------------------
# 33. TEST PREDICTION AFTER TUNING
# ------------------------------------------------------------

y_pred = best_tree.predict(X_test)


# ------------------------------------------------------------
# 34. CLASSIFICATION REPORT AFTER TUNING
# ------------------------------------------------------------

print(
    "\n========== CLASSIFICATION REPORT "
    "AFTER TUNING =========="
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Accuracy

print(
    "Accuracy =",
    accuracy_score(y_test, y_pred)
)


# ------------------------------------------------------------
# 35. TRAINING AND TESTING ACCURACY
# ------------------------------------------------------------

train_accuracy = best_tree.score(
    X_train,
    y_train
)

test_accuracy = best_tree.score(
    X_test,
    y_test
)

print(
    "\nTraining Accuracy =",
    train_accuracy
)

print(
    "Testing Accuracy =",
    test_accuracy
)


# ------------------------------------------------------------
# 36. VISUALIZE TUNED DECISION TREE
# ------------------------------------------------------------

plt.figure(figsize=(25, 15))

plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=[
        str(c)
        for c in best_tree.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree after Hyperparameter Tuning"
)

plt.show()


# ------------------------------------------------------------
# 37. CROSS VALIDATION
# ------------------------------------------------------------

cv_scores = cross_val_score(
    best_tree,
    X_train,
    y_train,
    cv=10,
    scoring="accuracy"
)

print(
    "\n========== CROSS VALIDATION =========="
)

print(
    "Cross-Validation Scores:",
    cv_scores
)

print(
    "Mean CV Accuracy:",
    cv_scores.mean()
)

print(
    "Standard Deviation:",
    cv_scores.std()
)


# ------------------------------------------------------------
# 38. LEARNING CURVE
# ------------------------------------------------------------

train_sizes, train_scores, test_scores = learning_curve(
    best_tree,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_scores.mean(axis=1),
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    test_scores.mean(axis=1),
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# 39. PREDICT NEW WINE QUALITY
# ------------------------------------------------------------

new_wine = pd.DataFrame([
    {
        "fixed acidity": 7.0,
        "volatile acidity": 0.80,
        "citric acid": 0.20,
        "residual sugar": 9.0,
        "chlorides": 0.09,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 90.0,
        "density": 0.996,
        "pH": 3.3,
        "sulphates": 0.60,
        "alcohol": 50.0
    }
])


# ------------------------------------------------------------
# 40. PREDICT WINE QUALITY
# ------------------------------------------------------------

prediction = best_tree.predict(
    new_wine
)

print(
    "\n========== NEW WINE PREDICTION =========="
)

print(
    "Predicted Wine Quality:",
    prediction[0]
)


# ------------------------------------------------------------
# 41. PREDICTION PROBABILITY
# ------------------------------------------------------------

probability = best_tree.predict_proba(
    new_wine
)

print(
    "Predicted Quality:",
    prediction[0]
)

print(
    "Prediction Probabilities:",
    probability
)


# ------------------------------------------------------------
# END OF ML LAB 4
# ------------------------------------------------------------

print("\n========== PROGRAM COMPLETED ==========")