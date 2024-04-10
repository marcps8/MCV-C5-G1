import pickle as pkl
import sys
import os
import numpy as np
from network import NetworkImg2Text
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from triplets_dataset import TripletsDatasetImg2Txt, TripletsDatasetVal
from utils_week4 import get_triplets_from_image_to_text, generate_embeddings_img2txt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
    parser.add_argument("--model-name", default="img2txt_25perc_fasttext.pth", type=str)
    parser.add_argument("--mode", type=str, default="img2txt", choices=["text2img", "img2txt"])
    parser.add_argument("--text-model", type=str, default="fasttext", choices=["fasttext", "bert"])
    args = parser.parse_args()

    triplets_path = OUTPUT_PATH + f"/pickles/triplets_val.pkl"
    triplets_train_path = OUTPUT_PATH + f"/pickles/triplets_25perc_img2txt.pkl"
    text_embeddings_path = OUTPUT_PATH + f"/pickles/text_embeddings_train_25perc_img2txt_{args.text_model}.pkl"
    query_captions_path = OUTPUT_PATH + "/pickles" + f"/query_captions_img2txt_{args.text_model}.pkl"
    query_features_path = OUTPUT_PATH + "/pickles" + f"/query_features_img2txt_{args.text_model}.pkl"
    model_path = OUTPUT_PATH + f"/weights/{args.model_name}"
    config = {
        "out_path": OUTPUT_PATH,
        "embed_size": 300,
        "batch_size": 64,
        "text_model": args.text_model,
        "load_path": model_path
    }

    triplets_val = get_triplets_from_image_to_text(
        None, 
        load_triplets=True,
        output_path=triplets_path
    )

    triplets_train = get_triplets_from_image_to_text(
        None, 
        load_triplets=True,
        output_path=triplets_train_path
    )
    
    if os.path.exists(text_embeddings_path):
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = pkl.load(f)
    else:
        print("There are not training text embeddings.")
        sys.exit()
        
    path = "/ghome/group01/MCV-C5-G1/Week4/pickles/triplets_val.pkl"
    with open(path, "rb") as f:
        val_data = pkl.load(f)

    val_dataset = TripletsDatasetVal(triplets=triplets_val, root_dir=VAL_PATH, transform=get_transforms())
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=8)
    net = NetworkImg2Text(config, val_loader=val_data, text_embeddings=text_embeddings)
    net.load_model()
    if not (os.path.exists(query_captions_path) and os.path.exists(query_features_path)):
        net.retrieve(args.mode, save_path=OUTPUT_PATH + "/pickles")

    else:
        with open(query_features_path, "rb") as f:
            query_features = pkl.load(f)
        with open(query_captions_path, "rb") as f:
            query_captions = pkl.load(f)

    # Train NN
    print("Start NearestNeighbours training.")

    # Create a StandardScaler object
    scaler = StandardScaler()

    X_train = []
    Y_train = []
    X_train_caption = []
    X_train_ids = []
    for caption, feature in tqdm(text_embeddings.items()):
        X_train_caption.append(caption)
        X_train.append(net.embed.forward_text(feature))
        X_id = [triplet[0] for triplet in triplets_train if triplet[1] == caption]
        X_train_ids.append(X_id)
        # break # Per provar el retrieval. Borrar quan funcioni

    X_train = [tensor.flatten().cpu().detach().numpy() for tensor in X_train]
    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    print(X_train.shape)
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_train)

    # Evaluate on Validation dataset
    print("Start NearestNeighbours validation.")
    val_ids = []
    f_query_features = query_features.tolist()
    for i in range(query_captions):
        val_ids.append(val_data[i])
        # break # Per provar el retrieval. Borrar quan funcioni

    
    #f_query_features = [tensor.numpy().flatten().cpu().detach().numpy() for tensor in f_query_features]
    f_query_features = np.array(f_query_features)
    f_query_features = scaler.fit_transform(f_query_features)
    distances, indices = knn.kneighbors(f_query_features)

    results = {
        "train_id": indices,
        "train_captions": X_train_caption,
        "val_caption": query_captions,
        "val_id": val_ids
    }

    for i in range(10):
        print(f"Val ID: {results['val_id'][i]}")
        print(f"Val Caption: {results['val_caption'][i]}")
        print('-')
        print(f"Train ID: {results['train_id'][i]}")
        print("Possible captions: ")
        for j in range(len(results['train_id'][i])):
            print(f"Train Caption: {results['train_captions'][results['train_id'][i][j]]}")
        print("####################################")

    with open(OUTPUT_PATH + f"/pickles/results_emb300_img2txt_{args.text_model}", "wb") as f:
        pkl.dump(results, f)

    
    