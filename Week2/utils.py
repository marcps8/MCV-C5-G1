from pathlib import Path
from typing import List, Dict
from detectron2.structures import BoxMode
from PIL import Image
import pycocotools.mask as rletools
import numpy as np

class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_images(path: str) -> List[Dict]:
    output = []
    input_files = [x for x in Path(path).glob("*.png") if x.is_file()]
    input_files.sort(key=str)

    for ii, img_path in enumerate(input_files):
        img = Image.open(img_path)

        output.append(
            {
                "file_name": str(img_path),
                "height": img.height,
                "width": img.width,
                "image_id": ii,
            }
        )

    return output


def load_instances_txt(path: str):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, (
                    "Multiple objects with track id "
                    + fields[1]
                    + " in frame "
                    + fields[0]
                )
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                "size": [int(fields[3]), int(fields[4])],
                "counts": fields[5].encode(encoding="UTF-8"),
            }
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif (
                rletools.area(
                    rletools.merge(
                        [combined_mask_per_frame[frame], mask], intersect=True
                    )
                )
                > 0.0
            ):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask], intersect=False
                )
            objects_per_frame[frame].append(
                SegmentedObject(mask, class_id, int(fields[1]))
            )

    return objects_per_frame


def load_annotations(annotation_path: str):
    annotation = load_instances_txt(annotation_path)

    objs = {}
    for frame_id, objects in annotation.items():
        frame_objs = []
        for track in objects:
            if track.track_id != 10000:

                class_id = track.track_id // 1000  # or track.class_id
                instance_id = track.track_id % 1000

                bbox = rletools.toBbox(track.mask)

                coco_bbox = np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                )

                obj = {
                    "bbox": coco_bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                }
                frame_objs.append(obj)

        if len(frame_objs) != 0:
            objs[frame_id] = frame_objs

    return objs

if __name__ == "__main__":
    print(load_images("/ghome/group01/mcv/datasets/C5/KITTI-MOTS/instances/0000"))
    print(load_annotations("/ghome/group01/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt"))