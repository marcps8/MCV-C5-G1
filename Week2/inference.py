# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
from argparse import ArgumentParser

import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Folders
DATASET_DIR = "/export/home/group01/mcv/datasets/C5/KITTI-MOTS/{}/image_02/"
RESULTS_DIR = "results/{}/"
MODELS = {
    0: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    1: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
}

def inference(img_path, predictor, cfg):
    im = cv2.imread(img_path)
    outputs = predictor(im)

    # To visualize results on images
    visualizer = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0], scale=1.2)
    )
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def run_inference(model_index: int, mode: str):
    model = MODELS[model_index]
    model_name = model.split("/")[-1].split('.')[0]
    dataset_dir = DATASET_DIR.format(mode)
    results_dir = RESULTS_DIR.format(model_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)

    # Run inference with pre-trained Faster R-CNN (detection) and Mask R-CNN (detection and segmentation) on all KITTI-MOTS dataset
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        out_dir = os.path.join(results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        for image in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image)
            out_path = os.path.join(out_dir, image)
            img = inference(img_path, predictor, cfg)

            cv2.imwrite(out_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            print(f"Processed img {img_path} with model {model_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-index", "-n", type=int, help="Model yaml name", required=True
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Either training or testing",
        required=True,
        choices=["training", "testing"],
    )
    args = parser.parse_args()
    run_inference(args.model_index, args.mode)
