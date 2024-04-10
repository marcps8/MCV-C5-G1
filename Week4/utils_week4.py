import json
import os
import pickle
import random

import numpy as np
import torch
import tqdm
from PIL import Image
import fasttext
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract features from an image
def extract_features(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()


# Function to extract features from a folder of images
def extract_features_from_folder(folder_path, model, transform):
    features = []
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                features.append(extract_features(image_path, model, transform))
    return np.array(features), image_paths


# Function to save features into a pickle file
def save_features(features, file_path):
    os.makedirs(
        os.path.dirname(file_path), exist_ok=True
    )  # Create directory if it doesn't exist
    with open(file_path, "wb") as f:
        pickle.dump(features, f)


# Function to load features from a pickle file
def load_features(file_path):
    with open(file_path, "rb") as f:
        features = pickle.load(f)
    return features


def extract_labels(image_paths):
    labels = []
    for path in image_paths:
        label = os.path.basename(
            os.path.dirname(path)
        )  # Extract subfolder name as label
        labels.append(label)
    return labels


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0
    for i in range(len(predicted)):
        if actual == predicted[i]:
            score += 1

    return score / len(predicted)


def mapk(actual, predicted, k=10):
    pk_list = []
    for i in range(len(actual)):
        score = apk(actual[i], predicted[i], k)
        pk_list.append(score)
    return np.mean(pk_list)


def AP(actual, predicted):
    gtp = 0
    ap = 0
    for i in range(len(predicted)):
        a = apk(actual, predicted, i + 1)
        if actual == predicted[i]:
            b = 1
            gtp += 1
        else:
            b = 0
        c = a * b
        ap += c
    if gtp == 0:
        return 0
    return ap / gtp


def mAP(actual, predicted):
    ap_list = []
    for i in range(len(actual)):
        ap = AP(actual[i], predicted[i])
        ap_list.append(ap)
    return np.mean(ap_list)


def coco_annotations():
    with open(
        "/ghome/group01/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json", "r"
    ) as f:
        annotations = json.load(f)

    inverted_annotations = {}
    for category, category_data in annotations.items():
        if not (category in inverted_annotations.keys()):
            inverted_annotations[category] = {}
        for object_id, image_ids in category_data.items():
            for image_id in image_ids:
                if image_id in inverted_annotations[category].keys():
                    inverted_annotations[category][image_id].append(object_id)
                else:
                    inverted_annotations[category][image_id] = [object_id]

    with open("inverted_annotations/inverted_annotations.json", "w") as f:
        json.dump(inverted_annotations, f)


def get_triplets_from_text_to_image(
    annotations: dict = None, load_triplets: bool = False, output_path: str = None
):
    if load_triplets:
        with open(output_path, "rb") as f:
            triplets = pickle.load(f)

        return triplets

    triplets = []
    image_ids = list(set([annotation["image_id"] for annotation in annotations]))

    for annotation in tqdm.tqdm(annotations):
        anchor_caption = annotation["caption"]
        anchor_image_id = annotation["image_id"]

        negative_image_ids = [id for id in image_ids if id != anchor_image_id]
        negative_image_id = random.choice(negative_image_ids)

        triplets.append((anchor_caption, anchor_image_id, negative_image_id))

    with open(output_path, "wb") as f:
        pickle.dump(triplets, f)

    return triplets

# TO-DO
def get_triplets_from_image_to_text(
    annotations: dict = None, load_triplets: bool = False, output_path: str = None
):
    if load_triplets:
        with open(output_path, "rb") as f:
            triplets = pickle.load(f)

        return triplets

    triplets = []

    shuffle_annotations = annotations
    for annotation in tqdm.tqdm(annotations):
        anchor_caption = annotation["caption"]
        anchor_image_id = annotation["image_id"]

        min_similarity = 1
        most_negative_caption = None
        random.shuffle(shuffle_annotations)
        for annotation in shuffle_annotations:
            if annotation["image_id"] == anchor_image_id:
                continue
        
            negative_caption = annotation["caption"]
            similarity_score = calculate_similarity(anchor_caption, negative_caption)
            if similarity_score < min_similarity:
                min_similarity = similarity_score
                most_negative_caption = negative_caption

                if similarity_score > 0.2 and similarity_score < 0.5:
                    break

        print(f"Posit,ve Sentence: {anchor_caption}")
        print(f"Negative Sentence: {most_negative_caption}")
        print(f"Similarity Score: {similarity_score}")
        triplets.append((anchor_image_id, anchor_caption, most_negative_caption))

    with open(output_path, "wb") as f:
        pickle.dump(triplets, f)

    return triplets


def get_correspondences_from_image_to_text(annotations:dict, load_correspondences, output_path):

    
    correspondences = []
    
    if load_correspondences:
        with open(output_path, "rb") as f:
            corresp = pickle.load(f)

        return corresp        
    
    for annotation in tqdm.tqdm(annotations):
        anchor_caption = annotation["caption"]
        anchor_image_id = annotation["image_id"]
        
        correspondences.append((anchor_image_id, anchor_caption))
    
    
    with open(output_path, "wb") as f:
        pickle.dump(correspondences, f) 
        
    return correspondences
    
    


def calculate_similarity(sentence1, sentence2):
    vectorizer = CountVectorizer()
    vectorized_sentences = vectorizer.fit_transform([sentence1, sentence2])

    cosine_sim = cosine_similarity(vectorized_sentences)
    similarity_score = cosine_sim[0][1]

    return similarity_score


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def embeddings_fasttext(model_txt, captions):
    embedded_captions = {}
    for caption in tqdm.tqdm(captions):
        embedded_word = []
        for word in caption.split():
            if word.lower() in model_txt:
                embedded_word.append(torch.tensor(model_txt[word.lower()]))
        embedded_captions.update({caption: torch.stack(embedded_word).mean(dim=0)})
    return embedded_captions


def embeddings_bert(model_txt, tokenizer, captions):
    embedded_captions = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_txt.to(device)
    for caption in tqdm.tqdm(captions):
        inputs = tokenizer(
            caption,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
            return_attention_mask=True,
        ).to(device)
        embedded_caption = model_txt(**inputs).last_hidden_state[:, 0, :]
        embedded_captions.update({caption: embedded_caption.squeeze()})
    return embedded_captions


def generate_embeddings(triplets, text_model):
    captions = [triplet[0] for triplet in triplets]
    if text_model == "fasttext":
        model_txt = fasttext.load_model("/export/home/mcv/C5/fasttext_wiki.en.bin")
        embeddings = embeddings_fasttext(model_txt, captions=captions)
    elif text_model == "bert":
        model_txt = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_txt.eval()
        for param in model_txt.parameters():
            param.requires_grad = False

        embeddings = embeddings_bert(model_txt, tokenizer, captions=captions)

    return embeddings

# Image to Text utils functions
def embeddings_fasttext_img2txt(model_txt, captions):
    embedded_captions = {}
    for caption_pair in tqdm.tqdm(captions):
        for caption in caption_pair:
            embedded_word = []
            for word in caption.split():
                if word.lower() in model_txt:
                    embedded_word.append(torch.tensor(model_txt[word.lower()]))
            embedded_captions.update({caption: torch.stack(embedded_word).mean(dim=0)})
    return embedded_captions


def embeddings_bert_img2txt(model_txt, tokenizer, captions):
    embedded_captions = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_txt.to(device)
    for caption_pair in tqdm.tqdm(captions):
        for caption in caption_pair:
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=True,
                return_attention_mask=True,
            ).to(device)
            embedded_caption = model_txt(**inputs).last_hidden_state[:, 0, :]
            embedded_captions.update({caption: embedded_caption.squeeze().cpu()})
    return embedded_captions


def generate_embeddings_img2txt(triplets, text_model):
    captions = [triplet[1:] for triplet in triplets]
    if text_model == "fasttext":
        model_txt = fasttext.load_model("/export/home/mcv/C5/fasttext_wiki.en.bin")
        embeddings = embeddings_fasttext_img2txt(model_txt, captions=captions)
    elif text_model == "bert":
        model_txt = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_txt.eval()
        for param in model_txt.parameters():
            param.requires_grad = False

        embeddings = embeddings_bert_img2txt(model_txt, tokenizer, captions=captions)

    return embeddings

