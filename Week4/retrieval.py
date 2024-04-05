from network import Network
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from triplets_dataset import TripletsDataset
from utils_week4 import get_triplets_from_text_to_image, generate_embeddings

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week4"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"

def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text-model", type=str, default="fasttext", choices=["fasttext", "bert"])
    args = parser.parse_args()

    triplets_path = OUTPUT_PATH + f"/pickles/triplets_val.pkl"
    model_path = OUTPUT_PATH + f"/weights/text2img_25perc_{args.text_model}.pth"
    config = {
        "out_path": OUTPUT_PATH,
        "embed_size": 32,
        "batch_size": 64,
        "text_model": args.text_model,
        "load_path": model_path
    }

    triplets = get_triplets_from_text_to_image(
        None, 
        load_triplets=True,
        output_path=triplets_path
    )
    
    text_embeddings = generate_embeddings(triplets, config["text_model"])
    val_dataset = TripletsDataset(triplets=triplets, root_dir=VAL_PATH, transform=get_transforms())
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    net = Network(config, val_loader=val_loader, text_embeddings=text_embeddings)
    net.load_model()
    net.retrieve(save_path=OUTPUT_PATH + "/pickles")