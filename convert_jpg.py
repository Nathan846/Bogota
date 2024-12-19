import cv2
import os

# Input and output directories
input_dir = "unet_data/labels/"
output_dir = "unet_data/jpg_labels"
os.makedirs(output_dir, exist_ok=True)

# Iterate through .tif files and convert to .jpg
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")

        # Read the .tif image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Write as .jpg
            cv2.imwrite(output_path, img)
            print(f"Converted: {input_path} to {output_path}")
        else:
            print(f"Failed to read: {input_path}")
