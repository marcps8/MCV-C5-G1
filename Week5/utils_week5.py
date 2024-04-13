import json
import os
import pickle
import random
import copy

import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


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


# Function to get caption for a given image_id
def get_captions_dict(captions):
    dict = {}
    for item in captions:
        dict[item["image_id"]] = item["caption"]
    return dict  


def get_triplets_from_text_to_image(
    annotations: dict = None, load_triplets: bool = False, output_path: str = None
):
    MAX_NEGATIVE_ITERATIONS = 100
    if load_triplets:
        with open(output_path, "rb") as f:
            triplets = pickle.load(f)

        return triplets
    captions_dict = get_captions_dict(annotations)
    triplets = []
    image_ids = list(set([annotation["image_id"] for annotation in annotations]))
    for annotation in tqdm.tqdm(annotations):
        anchor_caption = annotation["caption"]
        anchor_image_id = annotation["image_id"]

        negative_image_ids = [id for id in image_ids if id != anchor_image_id]

        # Try to get a soft negative: caption must be the
        # less similar as possible in positive and negative images

        negative_image_id = random.choice(negative_image_ids)
        min_distance_to_anchor = calculate_similarity(
            anchor_caption, captions_dict[negative_image_id]
        )
        for i in range(MAX_NEGATIVE_ITERATIONS):
            negative_candidate = random.choice(negative_image_ids)
            caption_candidate = captions_dict[negative_candidate]
            similarity_score = calculate_similarity(anchor_caption, caption_candidate)
            if similarity_score < min_distance_to_anchor:
                min_distance_to_anchor = similarity_score
                negative_image_id = negative_candidate

        triplets.append((anchor_caption, anchor_image_id, negative_image_id))

    with open(output_path, "wb") as f:
        pickle.dump(triplets, f)

    return triplets


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


def generate_embeddings(triplets):
    captions = [triplet[0] for triplet in triplets]
    model_txt = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_txt.eval()
    for param in model_txt.parameters():
        param.requires_grad = False

    embeddings = embeddings_bert(model_txt, tokenizer, captions=captions)

    return embeddings


def get_train_transforms():
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


def get_val_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_triplets_from_text_to_image_old(
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

def load_validation_image(root_dir, image_id):
    image_path = os.path.join(
        root_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
    )
    image = Image.open(image_path).convert("RGB")
    return image

def add_id_captions_json(json_file="new_captions.json"):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        
    for i, item in enumerate(json_data, start=1):
        item['id'] = i

    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=4)

def add_negative_id_captions_json(json_file="new_captions.json"):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    for i, item in enumerate(json_data, start=1):
        shuffled_data = copy.deepcopy(json_data)
        random.shuffle(shuffled_data)
        
        noun = item['keywords'][0]
        min_similarity = 1
        print(item['id'])
        for possible_negative in shuffled_data:
            if (item['id'] == possible_negative['id']) or (noun in possible_negative['keywords']):
                continue
            
            similarity_score = calculate_similarity(item['caption'], possible_negative['caption'])
            if similarity_score < min_similarity:
                min_similarity = similarity_score
                most_negative_id = possible_negative['id']
                
                if min_similarity < 0.1:
                    break
                
        item['negative_id'] = most_negative_id    

    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=4)
        
