import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# Configuration for the trained model
def setup_cfg(output_dir, num_classes=3):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if GPU is available
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")  # Path to your trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold if needed
    return cfg

# Load trained model
cfg = setup_cfg(output_dir="./output", num_classes=3)
predictor = DefaultPredictor(cfg)

# Input image for prediction
input_image_path = "unet_data/images_jpg/image_patch_1_t130.jpg"  # Update with your image path
output_mask_path = "output_mask.png"  # Path to save the predicted mask

# Read input image
input_image = cv2.imread(input_image_path)
if input_image is None:
    print(f"Could not read the input image: {input_image_path}")
else:
    # Make prediction
    predictions = predictor(input_image)

    # Get the predicted semantic segmentation mask
    pred_mask = predictions["sem_seg"].argmax(dim=0).cpu().numpy()

    # Save the mask
    cv2.imwrite(output_mask_path, pred_mask.astype(np.uint8) * 50)  # Multiply by 50 for better visualization
    print(f"Predicted mask saved to {output_mask_path}")
