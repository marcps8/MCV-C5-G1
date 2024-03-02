from ultralytics import YOLO
import os

from proves import get_KITTI_dataset_COCO_ids
from pathlib import Path
from pycocotools import mask as maskUtils

dataset_dir = Path("/export/home/group01/mcv/datasets/C5/KITTI-MOTS")

anns = get_KITTI_dataset_COCO_ids(dataset_dir, 'training')

print(anns[0])

"""
# Example annotation data
annotation_data = {
    'file_name': '/export/home/group01/mcv/datasets/C5/KITTI-MOTS/training/image_02/0011/000110.png',
    'height': 375,
    'width': 1242,
    'image_id': 1100110,
    'annotations': [
        {'bbox': [591.0, 178.0, 45.0, 41.0], 'category_id': 2},
        {'bbox': [21.0, 193.0, 181.0, 78.0], 'category_id': 2},
        # Include other annotations as needed
    ]
}
"""

# Define function to convert absolute bbox to YOLO format
def bbox_to_yolo(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    return x_center, y_center, width, height

"""
# Define YOLO class mapping (replace with your own class mapping)
class_mapping = {
    1: 0,
    2: 1
}
"""

# Convert annotations to COCO/YOLO format
yolo_annotations = []

annotation_data = anns[0]
for annotation in annotation_data['annotations']:
    bbox = annotation['bbox']
    category_id = annotation['category_id']

    # Convert absolute bbox to YOLO format
    x_center, y_center, width, height = bbox_to_yolo(bbox, annotation_data['width'], annotation_data['height'])
    
    # Map category ID to YOLO class ID
    yolo_class_id = category_id
    if yolo_class_id == -1:
        continue  # Skip if category ID is not mapped

    # Example RLE string
    rle_string = annotation['segmentation']['counts']
    print(rle_string)
    # Convert RLE string to mask

    # Append YOLO annotation
    yolo_annotations.append((yolo_class_id, x_center, y_center, width, height))


