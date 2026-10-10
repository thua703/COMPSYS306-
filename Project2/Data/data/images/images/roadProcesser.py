import os
from PIL import Image

# 当前脚本建议放在 images 文件夹里面运行
input_dir = "road"
output_dir = os.path.join("processedImages", "road")

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 支持的图片格式
valid_extensions = (".jpg", ".jpeg", ".png")

# 获取 road 文件夹中的所有图片，并按文件名排序
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith(valid_extensions)
])

print(f"Original road images: {len(image_files)}")

# 每 3 张取 1 张
selected_files = image_files[::3]

print(f"Selected road images: {len(selected_files)}")

# Resize size
target_size = (64, 48)

for filename in selected_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 打开图片
    with Image.open(input_path) as img:
        # 转为 RGB，避免某些图片格式问题
        img = img.convert("RGB")

        # Resize 到 64 × 48
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # 保存到 processedImages/road
        img.save(output_path)

print("Processing complete.")
print(f"Processed images saved to: {output_dir}")