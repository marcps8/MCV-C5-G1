import os
import cv2
import numpy as np
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

def class_mapping_kitti_to_coco(img_dir):

    seqmap_file = f'/ghome/group01/MCV-C5-G1/Week2/seqmaps/train.seqmap'
    imgs_anns, _ = load_seqmap(seqmap_file)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        print(img_dir)
        filename = os.path.join(img_dir, v["filename"])
        print(filename)
        break
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

BASE_DIR = "/ghome/group01/mcv/datasets/C5/KITTI-MOTS"
ann = class_mapping_kitti_to_coco(BASE_DIR + "/training")
