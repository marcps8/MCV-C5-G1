import json
import os
from pathlib import Path
import pandas as pd
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode
import glob



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

def class_mapping_kitti_to_coco(img_dir, mode: str):

    mapped_classes = {1: 2, 2: 0}  # Mapping KITTY to COCO classes

    if mode == "training":
       mode = "train"

    seqmap_file = f'/ghome/group01/MCV-C5-G1/Week2/seqmaps/{mode}.seqmap'
    imgs_anns, _ = load_seqmap(seqmap_file)

    print(imgs_anns)

    dataset_dicts = []
    id = 0
    for idx, v in enumerate(imgs_anns):
        record = {}
        instance_dir = os.path.join(img_dir, "instances_txt", v + ".txt")
        imgs_dir = os.path.join(img_dir, "training", "image_02", v)

        objects_per_frame = {}
        frames = []
        with open(instance_dir, "r") as f:
            for line in f:
                line = line.strip()
                fields = line.split(" ")

                frame = int(fields[0])
                track_id = int(fields[1])
                class_id = int(fields[2])

                if not (class_id == 1 or class_id == 2):
                    continue  

                height = int(fields[3]) 
                width = int(fields[4])

                # New image record
                if frame not in frames or len(frames)==0 and True:
                    if len(frames) != 0 and len(dataset_dicts) != 0:
                       dataset_dicts.append(record)

                    frames.append(frame)

                    for img in os.listdir(imgs_dir):
                        
                        image_frame = int(img.split('.')[0])
                        
                        if image_frame != frame:
                           continue
                        
                        record = {}
                        record["file_name"] = os.path.join(imgs_dir, img)
                        record["image_id"] = id
                        id += 1
                        record["height"] = height
                        record["width"] = width

                        mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode('utf8')}
                        bbox = toBbox(mask).tolist()

                        annotations = []
                        annotations.append({
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": mapped_classes[class_id],
                            "segmentation": mask,
                            "keypoints": [],
                            "iscrowd": 0
                        })

                        record["annotations"] = annotations

                        break

                # New object from the same image
                elif frame in frames:

                    record["annotations"].append({
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": mapped_classes[class_id],
                            "segmentation": mask,
                            "keypoints": [],
                            "iscrowd": 0
                        })

    return dataset_dicts           
                

BASE_DIR = "/ghome/group01/mcv/datasets/C5/KITTI-MOTS"
ann = class_mapping_kitti_to_coco(BASE_DIR, "training")
print(ann[0])

