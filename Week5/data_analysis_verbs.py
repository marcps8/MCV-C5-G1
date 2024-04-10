import os
import pickle as pkl
from collections import Counter, defaultdict

import nltk
from tqdm import tqdm
from utils_week4 import load_json

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5"
TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/train2014"
VAL_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
TRAIN_CAPTIONS_PATH = (
    "/export/home/group01/mcv/datasets/C5/COCO/captions_train2014.json"
)
VAL_CAPTIONS_PATH = "/export/home/group01/mcv/datasets/C5/COCO/captions_val2014.json"
PKL_VAL_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/val_noun_analysis.pkl"
)
PKL_VAL_VERB_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/val_verb_analysis.pkl"
)
PKL_TRAIN_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/train_analysis.pkl"
)

val_annotations = load_json(VAL_CAPTIONS_PATH)
train_annotations = load_json(TRAIN_CAPTIONS_PATH)
train_annotations = train_annotations["annotations"][0:100000]
val_annotations = val_annotations["annotations"]


def filter_tokens_with_verbs(sentences):
    noun_verb_mapping = defaultdict(list)
    noun_counts = Counter()
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence["caption"].lower())
        tagged_tokens = nltk.pos_tag(tokens)
        nouns = [
            token for token, pos in tagged_tokens if pos in ["NN", "NNS", "NNP", "NNPS"]
        ]
        verbs = [
            token for token, pos in tagged_tokens if pos in ["VB", "VBP", "VBZ", "VBG"]
        ]
        for noun in nouns:
            noun_verb_mapping[noun].extend(verbs)
            noun_counts[noun] += 1
    return noun_verb_mapping, noun_counts


# noun_verb_mapping, noun_counts = filter_tokens_with_verbs(sentences=sentences_)
# noun_verb_counts = {noun: Counter(verbs) for noun, verbs in noun_verb_mapping.items() if verbs}

if os.path.exists(PKL_VAL_ANALYSIS) and os.path.exists(PKL_VAL_VERB_ANALYSIS):
    with open(PKL_VAL_ANALYSIS, "rb") as f:
        val_noun_counts = pkl.load(f)
    with open(PKL_VAL_VERB_ANALYSIS, "rb") as f:
        val_verb_counts = pkl.load(f)
else:
    noun_verb_mapping, val_noun_counts = filter_tokens_with_verbs(val_annotations)
    val_verb_counts = {
        noun: Counter(verbs) for noun, verbs in noun_verb_mapping.items() if verbs
    }

    with open(PKL_VAL_ANALYSIS, "wb") as f:
        pkl.dump(val_noun_counts, f)

    with open(PKL_VAL_VERB_ANALYSIS, "wb") as f:
        pkl.dump(val_verb_counts, f)


print("Noun Counts:")
i = 0
val_noun_counts = {noun: count for noun, count in val_noun_counts.items() if count > 1}
for noun, count in val_noun_counts.items():
    print(f"{noun}: {count}")
    i += 1
    if i >= 20:
        break

i = 0
print("\nNoun-Verb Counts:")
val_verb_counts = {
    noun: {verb: count for verb, count in verb_counter.items() if count >= 5}
    for noun, verb_counter in val_verb_counts.items()
}
for noun, verb_counter in val_verb_counts.items():
    print(f"{noun}: {verb_counter}")
    i += 1
    if i >= 20:
        break
