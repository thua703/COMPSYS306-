import os
from PIL import Image

# Run this script from the images directory
input_dir = "road"
output_dir = os.path.join("processedImages", "road")

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png")

# List all images in the road directory, sorted by filename
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith(valid_extensions)
])

print(f"Original road images: {len(image_files)}")

# Select every third image
selected_files = image_files[::3]

print(f"Selected road images: {len(selected_files)}")

# Resize size
target_size = (64, 48)

for filename in selected_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Open the image
    with Image.open(input_path) as img:
        # Convert to RGB to avoid image format issues
        img = img.convert("RGB")

        # Resize to 64 x 48
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Save to processedImages/road
        img.save(output_path)

print("Processing complete.")
print(f"Processed images saved to: {output_dir}")