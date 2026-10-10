import os
import numpy as np
from PIL import Image

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataset_root = "dataset_split"
valid_extensions = (".jpg", ".jpeg", ".png")


def load_split(split_name):
    split_dir = os.path.join(dataset_root, split_name)

    X = []
    y = []
    paths = []

    class_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"\nLoading {split_name} set...")

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)

        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(valid_extensions)
        ])

        print(f"{class_name}: {len(files)}")

        for filename in files:
            image_path = os.path.join(class_dir, filename)

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_array = np.array(img, dtype=np.float32)
                img_array = img_array / 255.0
                features = img_array.flatten()

                X.append(features)
                y.append(class_name)
                paths.append(image_path)

    return np.array(X), np.array(y), paths


# ==============================
# Load training and validation data
# ==============================

X_train, y_train, train_paths = load_split("train")
X_val, y_val, val_paths = load_split("val")


print("\nTraining shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# ==============================
# Train baseline SVM
# ==============================

model = SVC(
    kernel="poly",
    degree=2,
    C=1,
    gamma="scale"
)

print("\nTraining SVM...")

model.fit(X_train, y_train)


# ==============================
# Validation
# ==============================

y_val_pred = model.predict(X_val)

print("\nMisclassified images:")

for i in range(len(y_val)):
    if y_val[i] != y_val_pred[i]:
        print(f"\nImage: {val_paths[i]}")
        print(f"True label: {y_val[i]}")
        print(f"Predicted: {y_val_pred[i]}")

val_accuracy = accuracy_score(y_val, y_val_pred)

print("\nValidation accuracy:", val_accuracy)

print("\nClassification report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_val, y_val_pred))