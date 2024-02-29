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

# Class configuration
KITTY_MOTS_CLASSES = {0: "Car", 1: "Pedestrian"}

def from_KITTY_to_COCO(path, part):
    COCO_classes = {0: 81, 1: 2, 2: 0, 10: 81}  # Mapping KITTY to COCO classes
    with open("./config/dataset_split.json") as f_splits:
        sequences = json.load(f_splits)[part]

    if part == "val":
        part = "training"
    sequence_dir = os.path.join(path, part, "image_02")
    annotations = []
    for seq in Path(sequence_dir).glob("*"):
        sequence = seq.parts[-1]
        if sequence not in sequences:
            continue
        with open(os.path.join(path, "instances_txt", sequence + ".txt")) as f_ann:
            gt = pd.read_table(
                f_ann,
                sep=" ",
                header=0,
                names=["frame", "obj_id", "class_id", "height", "width", "rle"],
                dtype={
                    "frame": int,
                    "obj_id": int,
                    "class_id": int,
                    "height": int,
                    "width": int,
                    "rle": str,
                },
            )
        for img_path in Path(seq).glob("*.png"):
            img_name = img_path.parts[-1]
            frame = int(img_path.parts[-1].split(".")[0])
            frame_gt = gt[gt["frame"] == frame]
            if len(frame_gt) == 0:
                continue
            ann = []
            for _, _, class_id, height, width, rle in frame_gt.itertuples(index=False):
                mask = {"counts": rle.encode("utf8"), "size": [height, width]}
                bbox = toBbox(mask).tolist()
                ann.append(
                    {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": COCO_classes[class_id],
                        "segmentation": mask,
                        "keypoints": [],
                        "iscrowd": 0,
                    }
                )
            annotations.append(
                {
                    "file_name": str(img_path),
                    "height": frame_gt.iloc[0]["height"],
                    "width": frame_gt.iloc[0]["width"],
                    "image_id": int(f"{sequence}{frame:05}"),
                    "sem_seg_file_name": str(
                        os.path.join(path, "instances", sequence, img_name)
                    ),
                    "annotations": ann,
                }
            )
    return annotations

def run_evaluation(model: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    predictor = DefaultPredictor(cfg)

    coco_names = [""] * 81
    coco_names[80] = "background"
    coco_names[0] = "pedestrian"
    coco_names[2] = "car"
    coco_names[71] = "sink"

    DATASET_NAME = "KITTI-MOTS-COCO_"
    for d in ["training", "val"]:
        DatasetCatalog.register(
            DATASET_NAME + d, lambda d=d: from_KITTY_to_COCO(BASE_DIR, d)
        )
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=coco_names, stuff_classes=coco_names
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