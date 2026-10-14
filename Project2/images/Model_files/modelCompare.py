import os
import time
import pickle
import numpy as np

from PIL import Image

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)



# Settings

DATASET_ROOT = "dataset_split"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

RANDOM_STATE = 42


# Load dataset

def load_split(split_name):
    split_dir = os.path.join(DATASET_ROOT, split_name)

    X = []
    y = []

    class_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f"\nLoading {split_name} set...")
    print("Classes:", class_names)

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)

        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ])

        print(f"{class_name}: {len(files)}")

        for filename in files:
            image_path = os.path.join(class_dir, filename)

            with Image.open(image_path) as img:

                # Make sure image is RGB
                img = img.convert("RGB")

                # Safety resize
                img = img.resize(
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    Image.Resampling.LANCZOS
                )

                # Convert to float and normalise to 0-1
                img_array = np.asarray(
                    img,
                    dtype=np.float32
                ) / 255.0

                # Flatten 64 x 48 x 3 -> 9216 features
                X.append(img_array.flatten())
                y.append(class_name)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    return X, y



# Evaluate one model

def evaluate_model(name, model, X_train, y_train, X_val, y_val):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start = time.perf_counter()

    y_pred = model.predict(X_val)

    inference_time = time.perf_counter() - start

    inference_ms_per_image = (
        inference_time / len(X_val)
    ) * 1000.0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    model_bytes = pickle.dumps(model)

    model_size_mb = len(model_bytes) / (1024 * 1024)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"\nValidation accuracy:      {accuracy:.4f}")
    print(f"Macro precision:          {precision:.4f}")
    print(f"Macro recall:             {recall:.4f}")
    print(f"Macro F1:                 {f1:.4f}")

    print(f"\nTraining time:            {train_time:.3f} s")
    print(
        f"Inference per image:      "
        f"{inference_ms_per_image:.3f} ms"
    )
    print(f"Model size:               {model_size_mb:.2f} MB")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            zero_division=0
        )
    )

    results = {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time": train_time,
        "inference_ms": inference_ms_per_image,
        "model_size_mb": model_size_mb,
    }

    return results



# Main

def main():

    print("=" * 70)
    print("SVM vs MLP Comparison")
    print("=" * 70)

    # --------------------------------------------------------
    # Load exactly the same training / validation data
    # --------------------------------------------------------

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("\nDataset shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = SVC(
        kernel="rbf",
        C=1,
        gamma="scale"
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    svm_results = evaluate_model(
        "RBF SVM",
        svm_model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    mlp_results = evaluate_model(
        "MLP",
        mlp_model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    header = (
        f"{'Model':<12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Train(s)':>12}"
        f"{'Infer(ms)':>12}"
        f"{'Size(MB)':>12}"
    )

    print(header)
    print("-" * 96)

    for result in [svm_results, mlp_results]:

        print(
            f"{result['model']:<12}"
            f"{result['accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
            f"{result['train_time']:>12.3f}"
            f"{result['inference_ms']:>12.3f}"
            f"{result['model_size_mb']:>12.2f}"
        )

    print("=" * 80)

    # Accuracy-based winner
    if svm_results["accuracy"] > mlp_results["accuracy"]:
        print("\nHigher validation accuracy: SVM")

    elif mlp_results["accuracy"] > svm_results["accuracy"]:
        print("\nHigher validation accuracy: MLP")

    else:
        print("\nValidation accuracy is tied.")

    print(
        "\nDo not choose the final model from accuracy alone. "
        "Also consider F1, inference speed, model size, "
        "and real-world robot performance."
    )


if __name__ == "__main__":
    main()