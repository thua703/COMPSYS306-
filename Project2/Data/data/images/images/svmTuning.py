import os
import numpy as np
from PIL import Image

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset_root = "dataset_split"
valid_extensions = (".jpg", ".jpeg", ".png")


def load_split(split_name):
    split_dir = os.path.join(dataset_root, split_name)

    X = []
    y = []

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

                # Normalize to 0~1
                img_array = img_array / 255.0

                # Flatten
                features = img_array.flatten()

                X.append(features)
                y.append(class_name)

    return np.array(X), np.array(y)


# ==============================
# Load train and validation sets
# ==============================

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")


print("\nTraining shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# ==============================
# Hyperparameter ranges
# ==============================

C_values = [0.1, 1, 10, 100]

gamma_values = [
    "scale",
    0.001,
    0.01,
    0.1
]


best_accuracy = 0
best_C = None
best_gamma = None

results = []


# ==============================
# Hyperparameter tuning
# ==============================

print("\nStarting hyperparameter tuning...\n")

for C in C_values:

    for gamma in gamma_values:

        print("----------------------------------")
        print(f"Testing C = {C}, gamma = {gamma}")

        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            y_val_pred
        )

        results.append(
            (C, gamma, accuracy)
        )

        print(
            f"Validation accuracy: "
            f"{accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_C = C
            best_gamma = gamma


# ==============================
# Print all results
# ==============================

print("\n\n===== ALL RESULTS =====")

for C, gamma, accuracy in results:

    print(
        f"C = {C:<5} "
        f"gamma = {str(gamma):<6} "
        f"accuracy = {accuracy:.4f}"
    )


# ==============================
# Best result
# ==============================

print("\n===== BEST MODEL =====")

print(f"Best C: {best_C}")
print(f"Best gamma: {best_gamma}")

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)