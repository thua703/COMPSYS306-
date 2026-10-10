import os
import shutil
from sklearn.model_selection import train_test_split

# 输入和输出文件夹
source_root = "processedImages"
output_root = "dataset_split"

# 数据集比例
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# 固定随机种子，保证每次运行结果一致
random_state = 42

valid_extensions = (".jpg", ".jpeg", ".png")

# 找到所有 class 文件夹
class_names = sorted([
    d for d in os.listdir(source_root)
    if os.path.isdir(os.path.join(source_root, d))
])

print("Classes found:")
for class_name in class_names:
    print(f"  {class_name}")

for class_name in class_names:
    class_dir = os.path.join(source_root, class_name)

    # 读取该 class 的所有图片
    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(valid_extensions)
    ])

    print(f"\n{class_name}: {len(files)} images")

    # 第一次 split:
    # 70% train, 30% temporary
    train_files, temp_files = train_test_split(
        files,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )

    # 第二次 split:
    # 将剩下 30% 平分成 15% validation 和 15% test
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

    # 复制文件
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