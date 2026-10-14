import os
import time
import pickle
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
    paths = []

    class_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)

        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(valid_extensions)
        ])

        for filename in files:
            image_path = os.path.join(class_dir, filename)

            with Image.open(image_path) as img:
                img = img.convert("RGB")

                img_array = np.array(
                    img,
                    dtype=np.float32
                ) / 255.0

                X.append(img_array.flatten())
                y.append(class_name)
                paths.append(image_path)

    return np.array(X), np.array(y), paths


X_train, y_train, _ = load_split("train")
X_val, y_val, val_paths = load_split("val")


models = [
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
        "RBF tuned",
        SVC(
            kernel="rbf",
            C=10,
            gamma=0.001
        )
    )
]


for model_name, model in models:

    print("\n================================")
    print(model_name)

    # Train
    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    # Predict
    start = time.perf_counter()

    y_pred = model.predict(X_val)

    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    # Support vectors
    total_support_vectors = len(
        model.support_vectors_
    )

    print(f"Accuracy: {accuracy:.4f}")

    print(
        f"Support vectors: "
        f"{total_support_vectors}"
    )

    print(
        f"Training time: "
        f"{training_time:.4f} s"
    )

    print(
        f"Validation prediction time: "
        f"{prediction_time:.4f} s"
    )

    print(
        f"Prediction time per image: "
        f"{prediction_time / len(X_val) * 1000:.4f} ms"
    )

    # Print misclassified images
    print("Misclassified images:")

    for i in range(len(y_val)):
        if y_val[i] != y_pred[i]:

            print(
                f"{val_paths[i]}"
                f" | True: {y_val[i]}"
                f" | Predicted: {y_pred[i]}"
            )

    # Save temporarily to compare model size
    filename = (
        model_name
        .replace(" ", "_")
        .lower()
        + ".pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(model, f)

    size_mb = os.path.getsize(filename) / (
        1024 * 1024
    )

    print(
        f"Model size: "
        f"{size_mb:.2f} MB"
    )