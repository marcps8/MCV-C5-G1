import random
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from network import Network
from triplets_dataset import TripletsDataset
from utils_week5 import (
    generate_embeddings,
    get_train_transforms,
    get_triplets_from_text_to_image_old as get_triplets_from_text_to_image,
    load_json,
)

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embed-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample-size", type=float, default=1.0)
    args = parser.parse_args()

    config = {
        "out_path": OUTPUT_PATH,
        "embed_size": args.embed_size,
        "batch_size": args.batch_size,
        "margin": 0.5,
        "lr": 1e-4,
    }

    triplets_path = OUTPUT_PATH + f"/pickles/triplets/triplets_final.pkl"
    model_path = OUTPUT_PATH + f"/weights/text2img_final"
    embed_path = OUTPUT_PATH + f"/weights/embed_final"

    train_annotations = load_json(TRAIN_CAPTIONS_PATH)
    len_train_annotations = len(train_annotations["annotations"])
    sampled_annotations = random.sample(
        train_annotations["annotations"], int(len_train_annotations * args.sample_size)
    )

    print("Processing triplets...")
    load_triplets = True
    triplets = get_triplets_from_text_to_image(
        sampled_annotations,
        load_triplets=load_triplets,
        output_path=triplets_path,
    )

    print("Generating embeddings...")
    text_embeddings = generate_embeddings(triplets)
    train_dataset = TripletsDataset(
        triplets=triplets, root_dir=TRAIN_PATH, transform=get_train_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8
    )

    net = Network(config, train_loader=train_loader, text_embeddings=text_embeddings)
    print("Training model...")
    net.train(args.epochs, save_embed_path=embed_path, save_model_path=model_path)
