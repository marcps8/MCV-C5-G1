import os
import cv2
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from network import Network
from triplets_dataset import TripletsDatasetVal
from utils_week5 import (
    generate_embeddings,
    get_val_transforms,
    get_triplets_from_text_to_image_old as get_triplets_from_text_to_image,
    load_json,
    load_validation_image
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
    triplets_val
):
    print("Start NearestNeighbours training.")
    scaler = StandardScaler()

    print("Dimension of db features: ", db_features.shape)
    print("Dimension of query features: ", query_features.shape) 

    X_train = np.array(db_features)
    X_train = scaler.fit_transform(X_train)
    print("X_train.shape: ", X_train.shape)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_train)

    print("Start NearestNeighbours validation.")
    X_test = np.array(query_features)
    X_test = scaler.transform(X_test)
    print("X_test.shape: ", X_test.shape)

    _, indices = knn.kneighbors(X_test)

    print("Length of indices: ", len(indices))
    results = {}
    for idx in range(len(indices)):
        results.update({
            triplets_val[idx][0]: [
                triplets_val[img_idx][1] for img_idx in indices[idx]
            ]
        })
        
    caption = triplets_val[0][0]    
    retrieved_images = results[caption]
    print("Caption: ", caption)
    print("Retrieved: ", retrieved_images)
    for img in retrieved_images:
        image = np.array(load_validation_image(VAL_PATH, img))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_PATH + f"/retrieved_images/retrieved_images_{img}.png", image)
        
    with open(OUTPUT_PATH + f"/pickles/results/results_txt2img", "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embed-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="text2img_final.pth", type=str)
    parser.add_argument("--embed-name", default="embed_final.pth", type=str)
    args = parser.parse_args()

    triplets_path = OUTPUT_PATH + f"/pickles/triplets/triplets_val_final.pkl"

    text_embeddings_path = (
        OUTPUT_PATH + f"/pickles/text_embeddings/text_embeddings_val_final.pkl"
    )
    query_features_path = (
        OUTPUT_PATH + "/pickles/query_features" + f"/query_features_final.pkl"
    )
    db_features_path = OUTPUT_PATH + "/pickles/db_features" + f"/db_features_final.pkl"

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
            triplets_val = pkl.load(f)
        print("Exists triplets val with length ", len(triplets_val))
    else:
        val_annotations = load_json(VAL_CAPTIONS_PATH)
        triplets_val = get_triplets_from_text_to_image(
            val_annotations["annotations"],
            load_triplets=False,
            output_path=triplets_path,
        )

    if (
        os.path.exists(text_embeddings_path)
        and os.path.getsize(text_embeddings_path) > 0
    ):
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = dict(list(pkl.load(f).items()))
            print("Text embeddings validation loaded with size ", len(text_embeddings))
    else:
        with open(text_embeddings_path, "wb") as f:
            print("Generating embeddings val...")
            text_embeddings = generate_embeddings(triplets_val)
            print("Generated!")
            pkl.dump(text_embeddings, f)

    val_dataset = TripletsDatasetVal(
        triplets=triplets_val, root_dir=VAL_PATH, transform=get_val_transforms()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8
    )

    net = Network(
        config,
        val_loader=val_loader,
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
        triplets_val
    )
