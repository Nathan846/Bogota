import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator
import shutil


def get_segmentation_dicts(image_dir, label_dir, identifiers):
    dataset_dicts = []
    for idx, core_id in enumerate(identifiers):
        image_path = os.path.join(image_dir, f"image_patch_{core_id}.jpg")
        print(image_path)
        label_path = os.path.join(label_dir, f"label_patch_{core_id}.jpg")

        # Load image and label
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Assuming label is grayscale

        if img is None or label is None:
            print(f"Could not read the image or label: {image_path}, {label_path}")
            continue

        height, width = img.shape[:2]

        record = {
            "file_name": image_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "sem_seg_file_name": label_path,  # Path to the segmentation mask
        }
        dataset_dicts.append(record)


    print(f"Total records: {len(dataset_dicts)}")
    return dataset_dicts


# Directories for images and labels
image_dir = "unet_data/images_jpg/"
label_dir = "unet_data//labels_jpg/"

# Get image identifiers
image_filenames = os.listdir(image_dir)
identifiers = [filename.replace("image_patch_", "").replace(".jpg", "") for filename in image_filenames]

# Train-test split
train_identifiers, val_identifiers = train_test_split(identifiers, test_size=0.2, random_state=42)

# Register datasets
# Register datasets
DatasetCatalog.register("unet_train", lambda: get_segmentation_dicts(image_dir, label_dir, train_identifiers))
DatasetCatalog.register("unet_val", lambda: get_segmentation_dicts(image_dir, label_dir, val_identifiers))

# Add `ignore_label` to the metadata
MetadataCatalog.get("unet_train").set(stuff_classes=["background", "class1"], ignore_label=255)
MetadataCatalog.get("unet_val").set(stuff_classes=["background", "class1"], ignore_label=255)
# Configuration for Detectron2
cfg = get_cfg()

# Load a DeepLabV3+ model configuration for semantic segmentation
cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))

# Update configuration for semantic segmentation
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =3   # Adjust for the number of classes (including background)
cfg.INPUT.MASK_FORMAT = "bitmask"  # Ensure masks are in the correct format
# Set dataset
cfg.DATASETS.TRAIN = ("unet_train",)
cfg.DATASETS.TEST = ("unet_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# Model settings
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50  
cfg.MODEL.DEVICE = "cpu"

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Trainer with evaluation
class TrainerWithSemSegEvaluation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return SemSegEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

trainer = TrainerWithSemSegEvaluation(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Load the trained model for inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultTrainer(cfg)

# Save the trained model
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
save_path = "./saved_model/model_final.pth"
os.makedirs("./saved_model", exist_ok=True)
shutil.copy(trained_model_path, save_path)
print(f"Model saved to {save_path}")
