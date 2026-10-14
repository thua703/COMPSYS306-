import os
import shutil
from sklearn.model_selection import train_test_split

#define the input and output files
source_root = "processedImages_resized"
output_root = "dataset_split"

#ratio of split
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

random_state = 42

valid_extensions = (".jpg", ".jpeg", ".png")

# get all images
class_names = sorted([
    d for d in os.listdir(source_root)
    if os.path.isdir(os.path.join(source_root, d))
])

print("Classes found:")
for class_name in class_names:
    print(f"  {class_name}")

for class_name in class_names:
    class_dir = os.path.join(source_root, class_name)

    # read images in this class 
    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(valid_extensions)
    ])

    print(f"\n{class_name}: {len(files)} images")

    # first split:
    # 70% train, 30% temporary
    train_files, temp_files = train_test_split(
        files,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )

    # second split:
    # split 30% rest to 15% validation and 15% test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # copy the files
    for split_name, split_files in splits.items():
        output_dir = os.path.join(
            output_root,
            split_name,
            class_name
        )

        os.makedirs(output_dir, exist_ok=True)

        for filename in split_files:
            src = os.path.join(class_dir, filename)
            dst = os.path.join(output_dir, filename)

            shutil.copy2(src, dst)

    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")

print("\nDataset split complete.")
print(f"Output saved to: {output_root}")