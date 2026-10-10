import os
import numpy as np
from PIL import Image

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


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

                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                # Normalize
                img_array = img_array / 255.0

                # Flatten
                features = img_array.flatten()

                X.append(features)
                y.append(class_name)
                paths.append(image_path)

    return np.array(X), np.array(y), paths


# ==============================
# Load train and test data
# ==============================

X_train, y_train, _ = load_split("train")
X_test, y_test, test_paths = load_split("test")

print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ==============================
# Final selected SVM
# ==============================

model = SVC(
    kernel="poly",
    degree=2,
    C=1,
    gamma="scale"
)


# ==============================
# Train final selected model
# ==============================

print("\nTraining final SVM...")

model.fit(X_train, y_train)


# ==============================
# Test
# ==============================

print("\nTesting final SVM...")

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


print("\n===== TEST RESULTS =====")

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_test_pred
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)


# ==============================
# Misclassified images
# ==============================

print("\nMisclassified images:")

error_count = 0

for i in range(len(y_test)):

    if y_test[i] != y_test_pred[i]:

        error_count += 1

        print(
            f"{test_paths[i]}"
            f" | True: {y_test[i]}"
            f" | Predicted: {y_test_pred[i]}"
        )

print(
    f"\nTotal misclassified: "
    f"{error_count} / {len(y_test)}"
)