import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Load extracted features
# -----------------------------

data_path = "results/emotion_features.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -----------------------------
# 2. Separate features/labels
# -----------------------------

X = df.drop("emotion", axis=1)
y = df["emotion"]


# -----------------------------
# 3. Encode emotion labels
# -----------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# -----------------------------
# 4. Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 5. Scale features
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 6. Create neural network
# -----------------------------

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)


# -----------------------------
# 7. Train model
# -----------------------------

print("\nTraining model...")

model.fit(X_train_scaled, y_train)

print("Training completed!")


# -----------------------------
# 8. Make predictions
# -----------------------------

y_pred = model.predict(X_test_scaled)


# -----------------------------
# 9. Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# -----------------------------
# 10. Classification report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# -----------------------------
# 11. Save model
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/emotion_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

print("\nModels saved successfully!")


# -----------------------------
# 12. Confusion matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Speech Emotion Recognition - Confusion Matrix")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")

plt.close()

print("Confusion matrix saved!")


# -----------------------------
# 13. Save accuracy
# -----------------------------

with open("results/model_accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Accuracy saved!")
print("\nPROJECT MODEL TRAINING COMPLETED!")