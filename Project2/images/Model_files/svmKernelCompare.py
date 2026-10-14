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
# Models to compare
# ==============================

models = [
    (
        "Linear",
        SVC(
            kernel="linear",
            C=1
        )
    ),

    (
        "Polynomial degree 2",
        SVC(
            kernel="poly",
            degree=2,
            C=1,
            gamma="scale"
        )
    ),

    (
        "Polynomial degree 3",
        SVC(
            kernel="poly",
            degree=3,
            C=1,
            gamma="scale"
        )
    ),

    (
        "RBF baseline",
        SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        )
    ),

    (
        "RBF tuned",
        SVC(
            kernel="rbf",
            C=10,
            gamma=0.001
        )
    )
]


# ==============================
# Compare models
# ==============================

results = []

print("\nStarting kernel comparison...\n")

for model_name, model in models:

    print("----------------------------------")
    print(f"Training: {model_name}")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    results.append(
        (model_name, accuracy)
    )

    print(
        f"Validation accuracy: "
        f"{accuracy:.4f}"
    )


# ==============================
# Print results
# ==============================

print("\n===== KERNEL COMPARISON =====")

for model_name, accuracy in results:

    print(
        f"{model_name:<25} "
        f"accuracy = {accuracy:.4f}"
    )


# ==============================
# Best model
# ==============================

best_model = max(
    results,
    key=lambda x: x[1]
)

print("\n===== BEST MODEL =====")

print(f"Model: {best_model[0]}")
print(
    f"Validation accuracy: "
    f"{best_model[1]:.4f}"
)