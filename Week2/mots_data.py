import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os


def load_txt(path):
    objects_per_frame = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            track_id = int(fields[1])
            class_id = int(fields[2])  
            if not(class_id == 1 or class_id == 2):          
                mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode('utf8')}
                
            gt = [frame, track_id]

            objects_per_frame[frame].append(
            mask,
            class_id,
            int(fields[1])
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


if __name__ == '__main__':
    seqmap_file = '/ghome/group01/MCV-C5-G1/Week2/seqmaps/train.seqmap'
    seqmap, max_frames = load_seqmap(seqmap_file)
    print(seqmap)
    txt_path = '/ghome/group01/mcv/datasets/C5/KITTI-MOTS/instances_txt/0000.txt'
    objects = load_txt(txt_path)

    print(objects[0][0].track_id)
    print(objects[0][0].mask['counts'])

