import random
import utils_week4 as utils
from triplets_dataset import CorrespondencesDatasetImg2Txt
from torch.utils.data import DataLoader
from network import NetworkImg2Text
from torchvision import transforms
from argparse import ArgumentParser
import pickle
import os 
import torch
import io

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week4"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
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
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--text-model", type=str, default="fasttext", choices=["fasttext", "bert"])
    parser.add_argument("--sample-size", type=float, default=0.25)
    parser.add_argument("--online_mining", type=bool, default=True)
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

    corresp_path = OUTPUT_PATH + f"/pickles/correspondences_25perc_img2txt.pkl"
    model_path = OUTPUT_PATH + f"/weights/text2img_25perc_img2txt_online_angular_{args.text_model}.pth"
    text_embeddings_path = OUTPUT_PATH + f"/pickles/text_embeddings_train_25perc_img2txt_{args.text_model}_online.pkl"
    bert_text_embeddings_path = OUTPUT_PATH
    
    train_annotations = utils.load_json(TRAIN_CAPTIONS_PATH)
    val_annotations = utils.load_json(VAL_CAPTIONS_PATH)
    len_train_annotations = len(train_annotations["annotations"])
    
    sampled_annotations = random.sample(
        train_annotations["annotations"], 
        int(len_train_annotations * args.sample_size) # If 1.0, it won't be sampled
    )

    print("Processing correspondences...")


    load_correspondences = True
    correspondences = utils.get_correspondences_from_image_to_text(sampled_annotations,
                                                                   load_correspondences,
                                                                   output_path=corresp_path)
    
    if os.path.exists(text_embeddings_path):
        print("Extracting the embeddigns")
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = pickle.load(f)
                        
    else:
        print("Generating embeddings...")
        text_embeddings = utils.generate_embeddings_img2txt(correspondences, config["text_model"])
        with open(text_embeddings_path, "wb") as f:
            pickle.dump(text_embeddings, f)
    

    train_dataset = CorrespondencesDatasetImg2Txt(triplets=correspondences, root_dir=TRAIN_PATH, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    net = NetworkImg2Text(config, train_loader=train_loader, text_embeddings=text_embeddings)
    print("Training model...")
    net.train_online(args.epochs, save_path=model_path)