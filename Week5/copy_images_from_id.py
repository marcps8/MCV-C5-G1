import os
import shutil

import numpy as np

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5/retrieval"
# Ensure to remove all the images and files from the output folder
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
ids = [419074, 203564, 203564, 203564, 459912]
#ids = [item for sublist in ids for item in sublist]
for id in ids:
    padded_id = str(id).zfill(6)  # Pad the id with zeros to ensure it has 6 digits
    filename = "COCO_val2014_000000" + padded_id + ".jpg"
    source_path = os.path.join(TRAIN_PATH, filename)
    destination_path = os.path.join(OUTPUT_PATH, filename)
    shutil.copy(source_path, destination_path)
