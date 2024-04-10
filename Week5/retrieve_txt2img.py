import os
import pickle as pkl
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import Network
from triplets_dataset import TripletsDataset, TripletsDatasetVal
from utils_week5 import (
    generate_embeddings,
    get_val_transforms,
    get_train_transforms,
    get_triplets_from_text_to_image,
    load_json,
)

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)


def retrieval(
    db_features,
    query_features,
    db_text_embeddings,
    text_embeddings,
    triplets_train,
    triplets_val,
):
    # Train NN
    print("Start NearestNeighbours training.")

    # Create a StandardScaler object
    scaler = StandardScaler()

    X_train = []
    Y_train = []
    X_train_caption = []
    X_train_ids = []
    idx = 0
    print("Dimension of all db feature:::::::", len(db_features))
    print("Dimension of all query feature:::::::", len(query_features))
    db_text_embeddings_values = list(db_text_embeddings.values())
    for feature in db_features:
        caption = db_text_embeddings_values[idx]
        X_train_caption.append(caption)
        feature_tensor = torch.tensor(
            feature, device=net.device
        )  # Convert numpy array to PyTorch tensor
        X_train.append(feature_tensor)
        X_id = [triplet[1] for triplet in triplets_train if triplet[0] == caption]
        X_train_ids.append(X_id)
        idx += 1

    X_train = [tensor.flatten().cpu().detach().numpy() for tensor in X_train]
    X_train = np.array(X_train)
    X_train = scaler.fit_transform(X_train)
    print("X_train.shape : ", X_train.shape)
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_train)

    # Evaluate on Validation dataset
    # query features are text features
    # need to store also id
    # X train are text features
    print("Start NearestNeighbours validation.")
    val_ids = []
    query_captions = []
    f_query_features = []
    i = 0

    print(len(text_embeddings.items()))
    print(len(triplets_val))
    for caption, feature in tqdm(text_embeddings.items()):
        val_id = [triplet[1] for triplet in triplets_val if triplet[0] == caption]
        # id of the image for that caption in queries
        val_ids.append(val_id)
        query_captions.append(caption)
        f_query_features.append(net.embed(feature))
        i += 1

    f_query_features = [
        tensor.flatten().cpu().detach().numpy() for tensor in f_query_features
    ]
    f_query_features = np.array(f_query_features)
    f_query_features = scaler.fit_transform(f_query_features)
    _, indices = knn.kneighbors(f_query_features)
    print("Length of indices: ", len(indices))
    results = {
        "train_id": indices,
        "train_captions": X_train_caption,
        "val_caption": query_captions,
        "val_id": val_ids,
    }

    for i in range(len(indices)):
        print(f"Train ID: {results['train_id'][i]}")
        ids = results["train_id"][i]
        real_ids = [X_train_ids[x] for x in ids]
        print(f"Train ID(real IDs): {real_ids}")
        print(
            f"Train Caption(first): {results['train_captions'][results['train_id'][i][0]]}"
        )

        print(f"Val ID: {results['val_id'][i]}")
        print(f"Val Caption: {results['val_caption'][i]}")
        print("####################################")

    print("Text embeddings database loaded with size ", len(db_text_embeddings))
    print("Text embeddings query loaded with size ", len(text_embeddings))
    with open(OUTPUT_PATH + f"/pickles/results/results_img2txt", "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embed-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="text2img.pth", type=str)
    parser.add_argument("--embed-name", default="embed.pth", type=str)
    args = parser.parse_args()

    triplets_path = OUTPUT_PATH + f"/pickles/triplets/triplets_val.pkl"
    triplets_train_path = OUTPUT_PATH + f"/pickles/triplets/triplets.pkl"

    db_text_embeddings_path = (
        OUTPUT_PATH + f"/pickles/text_embeddings/text_embeddings_train.pkl"
    )  # Changed to txt2img

    text_embeddings_path = (
        OUTPUT_PATH + f"/pickles/text_embeddings/text_embeddings_val.pkl"
    )  # Changed to txt2img

    # to do afegir embeddings val
    query_features_path = (
        OUTPUT_PATH + "/pickles/query_features" + f"/query_features.pkl"
    )
    db_features_path = OUTPUT_PATH + "/pickles/db_features" + f"/db_features.pkl"
    model_path = OUTPUT_PATH + f"/weights/{args.model_name}"
    embed_path = OUTPUT_PATH + f"/weights/{args.embed_name}"
    config = {
        "out_path": OUTPUT_PATH,
        "model_path": model_path,
        "embed_path": embed_path,
        "embed_size": args.embed_size,
        "batch_size": args.batch_size,
        "margin": 0.5,
        "lr": 1e-4,
    }

    if os.path.exists(triplets_path):
        with open(triplets_path, "rb") as f:
            triplets_val = pkl.load(f)[:100]
        print("Exists triplets val with length ", len(triplets_val))
    else:
        val_annotations = load_json(VAL_CAPTIONS_PATH)
        triplets_val = get_triplets_from_text_to_image(
            val_annotations["annotations"][:100],
            load_triplets=False,
            output_path=triplets_path,
        )

    if os.path.exists(triplets_train_path):
        with open(triplets_train_path, "rb") as f:
            triplets_train = pkl.load(f)[:100]
        print("Exists triplets train with length ", len(triplets_train))
    else:
        train_annotations = load_json(TRAIN_CAPTIONS_PATH)
        triplets_train = get_triplets_from_text_to_image(
            train_annotations["annotations"][:100],
            load_triplets=False,
            output_path=triplets_train_path,
        )

    if (
        os.path.exists(text_embeddings_path)
        and os.path.getsize(text_embeddings_path) > 0
        and os.path.exists(db_text_embeddings_path)
        and os.path.getsize(db_text_embeddings_path) > 0
    ):
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = dict(list(pkl.load(f).items())[:100])
            print("Text embeddings validation loaded with size ", len(text_embeddings))
        with open(db_text_embeddings_path, "rb") as f:
            db_text_embeddings = dict(list(pkl.load(f).items())[:100])
            print("Text embeddings database loaded with size ", len(db_text_embeddings))
    else:
        # Save text_embeddings at the specified path
        with open(text_embeddings_path, "wb") as f:
            print("Generating embeddings val...")
            text_embeddings = generate_embeddings(triplets_val)
            print("Generated!")
            pkl.dump(text_embeddings, f)
        with open(db_text_embeddings_path, "wb") as f:
            print("Generating embeddings train...")
            db_text_embeddings = generate_embeddings(triplets_train)
            print("Generated!")
            pkl.dump(db_text_embeddings, f)

    val_dataset = TripletsDatasetVal(
        triplets=triplets_val, root_dir=VAL_PATH, transform=get_val_transforms()
    )
    train_dataset = TripletsDataset(
        triplets=triplets_train, root_dir=TRAIN_PATH, transform=get_train_transforms()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )
    net = Network(
        config,
        val_loader=val_loader,
        train_loader=train_loader,
        text_embeddings=text_embeddings,
    )
    net.load_model()

    if not (os.path.exists(db_features_path) and os.path.exists(query_features_path)):
        net.extract_features(
            save_path_db=db_features_path, save_path_query=query_features_path
        )
    with open(query_features_path, "rb") as f:
        query_features = pkl.load(f)
    with open(db_features_path, "rb") as f:
        db_features = pkl.load(f)

    retrieval(
        db_features,
        query_features,
        db_text_embeddings,
        text_embeddings,
        triplets_train,
        triplets_val,
    )
