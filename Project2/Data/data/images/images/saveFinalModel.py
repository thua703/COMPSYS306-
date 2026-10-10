import os
import pickle
import numpy as np
from PIL import Image

from sklearn.svm import SVC


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

                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                # Normalize 0~255 -> 0~1
                img_array = img_array / 255.0

                # Flatten 64x48x3 -> 9216 features
                features = img_array.flatten()

                X.append(features)
                y.append(class_name)

    return np.array(X), np.array(y)


# ==============================
# Load train + validation
# ==============================

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")

# Combine train and validation
X_final = np.concatenate(
    (X_train, X_val),
    axis=0
)

y_final = np.concatenate(
    (y_train, y_val),
    axis=0
)

print("\nFinal training dataset:")
print("X_final:", X_final.shape)
print("y_final:", y_final.shape)


# ==============================
# Final selected model
# ==============================

model = SVC(
    kernel="poly",
    degree=2,
    C=1,
    gamma="scale"
)


# ==============================
# Train final model
# ==============================

print("\nTraining final deployment model...")

model.fit(
    X_final,
    y_final
)

print("Training complete.")


# ==============================
# Save model
# ==============================

model_filename = "svm_final_model.pkl"

with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(
    f"\nFinal model saved as: "
    f"{model_filename}"
)

size_mb = os.path.getsize(
    model_filename
) / (1024 * 1024)

print(
    f"Model size: "
    f"{size_mb:.2f} MB"
)