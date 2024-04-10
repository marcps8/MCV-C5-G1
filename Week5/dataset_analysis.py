import os
import pickle as pkl
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm
from utils_week4 import generate_embeddings, get_triplets_from_text_to_image, load_json

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"
PKL_VAL_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/val_analysis.pkl"
)
PKL_TRAIN_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/train_analysis.pkl"
)

val_annotations = load_json(VAL_CAPTIONS_PATH)
train_annotations = load_json(TRAIN_CAPTIONS_PATH)
train_annotations = train_annotations["annotations"][0:100000]
val_annotations = val_annotations["annotations"]


def filter_tokens(tokens):
    filtered_tokens = []
    for token, pos in tqdm(nltk.pos_tag(tokens), desc="Filtering Tokens"):
        if pos in ["NN", "NNS", "NNP", "NNPS"]:
            filtered_tokens.append(token)
    return filtered_tokens


# tokens = [token for sentence in val_annotations for token in nltk.word_tokenize(sentence["caption"].lower())

if os.path.exists(PKL_VAL_ANALYSIS):
    with open(PKL_VAL_ANALYSIS, "rb") as f:
        val_token_counts = pkl.load(f)
else:
    tokens = []
    for annotation in tqdm(val_annotations, desc="Tokenizing validation captions"):
        caption_tokens = nltk.word_tokenize(annotation["caption"].lower())
        tokens.extend(caption_tokens)

    important_tokens = filter_tokens(tokens)
    val_token_counts = Counter(important_tokens)

    with open(PKL_VAL_ANALYSIS, "wb") as f:
        pkl.dump(val_token_counts, f)


if os.path.exists(PKL_TRAIN_ANALYSIS):
    with open(PKL_TRAIN_ANALYSIS, "rb") as f:
        train_token_counts = pkl.load(f)
else:
    tokens = []
    for annotation in tqdm(train_annotations, desc="Tokenizing training captions"):
        caption_tokens = nltk.word_tokenize(annotation["caption"].lower())
        tokens.extend(caption_tokens)

    important_tokens = filter_tokens(tokens)
    train_token_counts = Counter(important_tokens)

    with open(PKL_TRAIN_ANALYSIS, "wb") as f:
        pkl.dump(train_token_counts, f)

min_counts = 20
less_common_train = {
    token: count
    for token, count in train_token_counts.items()
    if count < val_token_counts[token] * 0.5 and val_token_counts[token] > min_counts
}
print(less_common_train)
print("Tokens less common in training compared to validation:")
for token, count in less_common_train.items():
    print(f"{token}: Training: {count} - Validation:  {val_token_counts[token]}")

"""
total_nouns = sum(token_counts.values())
noun_percentages = {noun: (count / total_nouns) * 100 for noun, count in token_counts.items()}

print(f"Most Common: {token_counts.most_common(20)}")
print(f"Less Common: {token_counts.most_common()[:-21:-1]}")
print(len(token_counts))

for noun, percentage in noun_percentages.items():
    print(f'{noun}: {percentage:.2f}%')

plt.figure(figsize=(10, 6))
plt.bar(token_counts.keys(), token_counts.values())
plt.xlabel('Token')
plt.ylabel('Frequency')
plt.title('Object Token Distribution in Dataset')
plt.xticks(rotation=90)
plt.savefig('plots/data_analysis/validation_analysis.jpg')

with open(PKL_ANALYSIS, "wb") as f:
            pkl.dump(token_counts, f)

"""
