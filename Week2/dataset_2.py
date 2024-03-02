import json
import os
from pathlib import Path
import pandas as pd
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode


def load_txt(path):
    objects_per_frame = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            track_id = int(fields[1])
            class_id = int(fields[2])  
            height = int(fields[3]) 
            width = int(fields[4]) 
         
            if not(class_id == 1 or class_id == 2):
              continue

            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode('utf8')}

            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            objects_per_frame[frame].append(
              (frame, track_id, class_id, height, width, mask)
            )

    return objects_per_frame


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

    sequence_dir = os.path.join(path, "training", "image_02")

    annotations = []
    for seq in Path(sequence_dir).glob("*"):
        sequence = seq.parts[-1]

        if sequence not in sequences:
            continue

        gt = load_txt(os.path.join(path, "instances_txt", sequence + ".txt"))

        for img_path in seq.glob("*.png"):
          img_name = img_path.parts[-1]
          frame = int(img_path.stem)  # Use stem to get the file name without extension
          print(gt[frame][0][0])
          break
          try:
            frame_gt = gt[frame]  # Retrieve ground truth for the current frame
          except:
            continue
          if not frame_gt:
              continue
          print(frame_gt)
          ann = []
          for _, _, class_id, height, width, mask in frame_gt:

              if class_id != 1 and class_id != 2:
                  continue

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
                  "height": height,
                  "width": width,
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
#print(ann[0])

