import os
import pandas as pd
import cv2
import torch
from argparse import ArgumentParser

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer


from dataset import class_mapping_kitti_to_coco
from inference import MODELS

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
PATH_RESULTS = "/ghome/group01/MCV-C5-G1/Week2/results/finetunning"
SAVE_PATH_TRAIN_INFERENCES_KM = "/ghome/group01/C5-W2/task_c/mask/train_inferences_KM"


# Class configuration
DATASET_NAME = "KITTI-MOTS-COCO_"

def inference(img_path, predictor, cfg):
    im = cv2.imread(img_path)
    outputs = predictor(im)

    # To visualize results on images
    visualizer = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    )
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def run_finetunning(model: str):
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

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("KITTI-MOTS-COCO_training",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16  
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 3000    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)  
    cfg.OUTPUT_DIR = PATH_RESULTS
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


    model_name = model.split("/")[-1].split('.')[0]
    cfg.MODEL.WEIGHTS = os.path.join(PATH_RESULTS, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # Model evaluation
    model_name = model.split("/")[-1].split('.')[0]
    output_dir = f"results/finetunning/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = COCOEvaluator("KITTI-MOTS-COCO_val", output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "KITTI-MOTS-COCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    dataset_dir = '/ghome/group01/mcv/datasets/C5/KITTI-MOTS/testing/image_02/'
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        out_dir = os.path.join(PATH_RESULTS, 'mask', folder)
        os.makedirs(out_dir, exist_ok=True)

        for image in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image)
            out_path = os.path.join(out_dir, image)
            img = inference(img_path, predictor, cfg)

            cv2.imwrite(out_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    """        
    im = cv2.imread(img_path)
    outputs = predictor(im)
    # To visualize results on images
    visualizer = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    )
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred_img = out.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(PATH_RESULTS, f'predicted_{model_name}.png'), pred_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    """


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-index", "-n", type=int, help="Model yaml name", required=True
    )
    args = parser.parse_args()
    model = MODELS[args.model_index]
    run_finetunning(model)