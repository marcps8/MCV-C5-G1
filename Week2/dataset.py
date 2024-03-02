import json
import os
from pathlib import Path
import pandas as pd
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode

def load_seqmap(seqmap_filename):
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      seq = "%04d" % int(fields[0])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
  return seqmap, max_frames

def class_mapping_kitti_to_coco(path, part):

    mapped_classes = {1: 2, 2: 0}  # Mapping KITTY to COCO classes
    if part == 'training':
        mode = 'train'
    else:
        mode = part
    seqmap_file = f'/ghome/group01/MCV-C5-G1/Week2/seqmaps/{mode}.seqmap'
    sequences, _ = load_seqmap(seqmap_file)
    """
    with open("./config/dataset_split.json") as f_splits:
        sequences = json.load(f_splits)[part]
        print(sequences)
    """

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

                if class_id != 1 and class_id != 2:
                    continue

                mask = {"counts": rle.encode("utf8"), "size": [height, width]}
                bbox = toBbox(mask).tolist()
                ann.append(
                    {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": mapped_classes[class_id],
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


BASE_DIR = "/ghome/group01/mcv/datasets/C5/KITTI-MOTS"
ann = class_mapping_kitti_to_coco(BASE_DIR, "training")
print(ann[0])
