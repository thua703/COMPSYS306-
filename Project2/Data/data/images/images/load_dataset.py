import os
import numpy as np
from PIL import Image

dataset_root = "dataset_split"

valid_extensions = (".jpg", ".jpeg", ".png")


def load_split(split_name):
    split_dir = os.path.join(dataset_root, split_name)

    X = []
    y = []

    # 获取所有 class
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
            if f.lower().endswith(valid_extensions)
        ])

        print(f"{class_name}: {len(files)}")

        for filename in files:
            image_path = os.path.join(class_dir, filename)

            with Image.open(image_path) as img:
                # 确保 RGB
                img = img.convert("RGB")

                # 转成 numpy array
                img_array = np.array(img, dtype=np.float32)

                # Normalize: 0~255 -> 0~1
                img_array = img_array / 255.0

                # Flatten
                features = img_array.flatten()

                X.append(features)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y


# 读取三个 dataset
X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)