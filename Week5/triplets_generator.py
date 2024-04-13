import random
from argparse import ArgumentParser

from utils_week5 import get_triplets_from_text_to_image, load_json

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-size", type=float, default=1.0)
    args = parser.parse_args()

    triplets_path = OUTPUT_PATH + f"/pickles/triplets/triplets_hard_negatives.pkl"
    train_annotations = load_json(TRAIN_CAPTIONS_PATH)
    len_train_annotations = len(train_annotations["annotations"])

    sampled_annotations = random.sample(
        train_annotations["annotations"],
        int(len_train_annotations * args.sample_size),
    )

    print("Processing triplets...")
    triplets = get_triplets_from_text_to_image(
        sampled_annotations, output_path=triplets_path
    )
    print("Example of triplets: ")
    print(triplets[:10])
    print("Total triplets: ", len(sampled_annotations))
