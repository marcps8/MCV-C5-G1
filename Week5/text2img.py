import random
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision import transforms

from network import Network
from triplets_dataset import TripletsDataset
from utils_week5 import generate_embeddings, get_triplets_from_text_to_image, load_json

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week4"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"


def get_transforms():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.3, hue=0.3),
            transforms.RandomResizedCrop(256, (0.15, 1.0)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embed-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--text-model", type=str, default="fasttext", choices=["fasttext", "bert"]
    )
    parser.add_argument("--sample-size", type=float, default=1.0)
    args = parser.parse_args()

    config = {
        "out_path": OUTPUT_PATH,
        "embed_size": args.embed_size,
        "batch_size": args.batch_size,
        "text_model": args.text_model,
        "margin": 0.5,
        "eps": 1e-7,
        "lr": 3e-4,
        "p": 2,
    }

    triplets_path = OUTPUT_PATH + f"/pickles/triplets_25perc_prova.pkl"
    model_path = OUTPUT_PATH + f"/weights/text2img_25perc_{args.text_model}.pth"

    train_annotations = load_json(TRAIN_CAPTIONS_PATH)
    val_annotations = load_json(VAL_CAPTIONS_PATH)
    len_train_annotations = len(train_annotations["annotations"])

    sampled_annotations = random.sample(
        train_annotations["annotations"],
        int(len_train_annotations * args.sample_size),  # If 1.0, it won't be sampled
    )

    print("Processing triplets...")
    load_triplets = False
    triplets = get_triplets_from_text_to_image(
        sampled_annotations, load_triplets=load_triplets, output_path=triplets_path
    )
    print("Example of triplets: ")
    print(triplets[:10])
    print("Total triplets: ", len(sampled_annotations))
