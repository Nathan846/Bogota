import cv2
import os
import random

# Input and output directories
image_input_dir = "unet_data/images/"
label_input_dir = "unet_data/labels/"

output_base_dir = "unet_data/split_data/"
train_image_dir = os.path.join(output_base_dir, "train/images")
train_label_dir = os.path.join(output_base_dir, "train/labels")
test_image_dir = os.path.join(output_base_dir, "test/images")
test_label_dir = os.path.join(output_base_dir, "test/labels")

# Create directories for train/test splits
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Train-test split ratio
split_ratio = 1.0

# Get list of image filenames and extract their core identifiers
image_filenames = [f for f in os.listdir(image_input_dir) if f.endswith(".tif")]
core_identifiers = [f.replace("image_", "").replace(".tif", "") for f in image_filenames]

# Shuffle and split the identifiers into train and test sets
random.shuffle(core_identifiers)
split_index = int(len(core_identifiers) * split_ratio)
train_identifiers = core_identifiers[:split_index]
test_identifiers = core_identifiers[split_index:]


def convert_and_save(input_dir, output_dir, core_ids, prefix, extension="jpg"):
    for core_id in core_ids:
        # Ensure prefix is applied correctly
        if core_id.startswith(prefix):
            input_filename = f"{core_id}.tif"
            output_filename = f"{core_id}.{extension}"
        else:
            input_filename = f"{prefix}{core_id}.tif"
            output_filename = f"{prefix}{core_id}.{extension}"

        input_path = os.path.join(input_dir, input_filename)
        output_path = os.path.join(output_dir, output_filename)

        # Read the .tif image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Raise an error if the image cannot be read
            raise FileNotFoundError(f"Failed to read image: {input_path}")
        
        # Write as .jpg
        success = cv2.imwrite(output_path, img)
        if not success:
            # Raise an error if the image cannot be written
            raise IOError(f"Failed to write image: {output_path}")
        
        print(f"Converted: {input_path} to {output_path}")

# Convert and save images and labels for training
convert_and_save(image_input_dir, train_image_dir, train_identifiers, prefix="image_")
convert_and_save(label_input_dir, train_label_dir, train_identifiers, prefix="label_")

# Convert and save images and labels for testing
convert_and_save(image_input_dir, test_image_dir, test_identifiers, prefix="image_")
convert_and_save(label_input_dir, test_label_dir, test_identifiers, prefix="label_")

print("Train-test split and conversion completed.")
