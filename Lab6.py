
# ============================================================
# LAB 6: Artificial Neural Network Classification
# Dataset: Seeds Dataset (seeds.csv)
# ============================================================

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load the Seeds dataset
df = pd.read_csv("seeds.csv")

print("First five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

# 3. Clean the dataset
# Remove duplicate rows and rows containing missing values
df = df.drop_duplicates()
df = df.dropna()

# Expected target column in the supplied PDF
target_column = "Class (1, 2, 3)"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found. "
        f"Available columns: {list(df.columns)}"
    )

# 4. Separate input features and target
X = df.drop(target_column, axis=1)
y_original = df[target_column].astype(int)

print("\nFeature shape:", X.shape)
print("Target shape:", y_original.shape)

print("\nOriginal classes:")
print(sorted(y_original.unique()))

# Convert classes 1, 2, 3 into 0, 1, 2 for ANN training
y = y_original - 1

print("\nEncoded classes:")
print(sorted(y.unique()))

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 6. Standardize the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Build the Artificial Neural Network
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

# 8. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nANN Model Summary:")
model.summary()

# 9. Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# 10. Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# 11. Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.title("ANN Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nTest loss:", test_loss)
print("Test accuracy:", test_accuracy)

# 13. Predict class probabilities
y_prob = model.predict(X_test_scaled, verbose=0)

print("\nFirst five prediction probabilities:")
print(y_prob[:5])

# 14. Convert probabilities to predicted class labels
y_pred = np.argmax(y_prob, axis=1)

print("\nFirst ten encoded predictions:")
print(y_pred[:10])

# Convert encoded labels back to original classes 1, 2, 3
y_pred_original = y_pred + 1
y_test_original = y_test.to_numpy() + 1

print("\nFirst ten predictions in original classes:")
print(y_pred_original[:10])

# 15. Compare actual and predicted classes
comparison = pd.DataFrame({
    "Actual class": y_test_original,
    "Predicted class": y_pred_original
})

print("\nActual vs Predicted:")
print(comparison.head(10))

# 16. Calculate evaluation metrics
accuracy = accuracy_score(
    y_test_original, y_pred_original
)

precision = precision_score(
    y_test_original,
    y_pred_original,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_original,
    y_pred_original,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test_original,
    y_pred_original,
    average="weighted",
    zero_division=0
)

print("\nEvaluation Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Display metrics in a table
metrics = pd.DataFrame({
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 score": [f1]
})

print("\nMetrics table:")
print(metrics.round(4))

# 17. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test_original,
        y_pred_original,
        labels=[1, 2, 3],
        zero_division=0
    )
)

# 18. Generate and display confusion matrix
cm = confusion_matrix(
    y_test_original,
    y_pred_original,
    labels=[1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[1, 2, 3],
    yticklabels=[1, 2, 3]
)

plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# 19. Save the trained model
model.save("lab6_ann_model.keras")

print("\nModel saved successfully as lab6_ann_model.keras")

# 20. Save evaluation metrics
metrics.to_csv("lab6_evaluation_metrics.csv", index=False)

print("Evaluation metrics saved successfully.")
print("\nLab 6 execution completed successfully.")
