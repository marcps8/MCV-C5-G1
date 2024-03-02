import json
import os
from pathlib import Path
import pandas as pd

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from pycocotools.mask import toBbox

from dataset import class_mapping_kitti_to_coco
from inference import MODELS
from argparse import ArgumentParser


# Set up logger
setup_logger()

# Environment configuration
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Path configuration
BASE_DIR = "/ghome/group01/mcv/datasets/C5/KITTI-MOTS"
PATH_TRAINING_SET = os.path.join(BASE_DIR, "training", "image_02")
PATH_TEST_SET = os.path.join(BASE_DIR, "testing", "image_02")
PATH_INSTANCES = os.path.join(BASE_DIR, "instances")
PATH_INSTANCES_TXT = os.path.join(BASE_DIR, "instances_txt")
SAVE_PATH_TRAIN_INFERENCES_KM = "/ghome/group01/C5-W2/task_c/mask/train_inferences_KM"


def run_evaluation(model: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    predictor = DefaultPredictor(cfg)

    class_labels = ["" for _ in range(81)]
    class_labels[80] = "background"
    class_labels[0] = "pedestrian"
    class_labels[2] = "car"

    DATASET_NAME = "KITTI-MOTS-COCO_"
    for d in ["training", "val"]:
        DatasetCatalog.register(
            DATASET_NAME + d, lambda d=d: class_mapping_kitti_to_coco(BASE_DIR, d)
        )
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=class_labels, stuff_classes=class_labels
        )
    
    model_name = model.split("/")[-1].split('.')[0]
    output_dir = f"results/evaluation/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Model evaluation
    print("Evaluating model")
    evaluator = COCOEvaluator(DATASET_NAME + "val", output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, DATASET_NAME + "val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-index", "-n", type=int, help="Model yaml name", required=True
    )
    args = parser.parse_args()
    model = MODELS[args.model_index]
    run_evaluation(model)