import os
import pickle as pkl
from collections import Counter, defaultdict
import numpy as np

import nltk
from tqdm import tqdm
from utils_week5 import load_json

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
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/train_noun_analysis.pkl"
)
PKL_TRAIN_VERB_ANALYSIS = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/data_analysis/train_verb_analysis.pkl"
)
TRIPLETS_DATASET = (
    "/ghome/group01/MCV-C5-G1/Week5/pickles/triplets/triplets.pkl"
)

def filter_tokens_with_verbs(sentences):
    noun_verb_mapping = defaultdict(list)
    noun_counts = Counter()
    for sentence in sentences:
        # tokens = nltk.word_tokenize(sentence["caption"].lower())
        tokens = nltk.word_tokenize(sentence.lower())
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


val_annotations = load_json(VAL_CAPTIONS_PATH)
train_annotations = load_json(TRAIN_CAPTIONS_PATH)

with open(TRIPLETS_DATASET, "rb") as f:
        train_annotations = pkl.load(f)
        
train_annotations = [caption for caption, _, _ in train_annotations]
val_annotations = val_annotations["annotations"]

# noun_verb_mapping, noun_counts = filter_tokens_with_verbs(sentences=sentences_)
# noun_verb_counts = {noun: Counter(verbs) for noun, verbs in noun_verb_mapping.items() if verbs}

if os.path.exists(PKL_TRAIN_ANALYSIS) and os.path.exists(PKL_TRAIN_VERB_ANALYSIS):
    with open(PKL_TRAIN_ANALYSIS, "rb") as f:
        train_noun_counts = pkl.load(f)
    with open(PKL_TRAIN_VERB_ANALYSIS, "rb") as f:
        train_verb_counts = pkl.load(f)
else:
    train_noun_verb_mapping, train_noun_counts = filter_tokens_with_verbs(train_annotations)
    train_verb_counts = {
        noun: Counter(verbs) for noun, verbs in train_noun_verb_mapping.items() if verbs
    }

    with open(PKL_TRAIN_ANALYSIS, "wb") as f:
        pkl.dump(train_noun_counts, f)

    with open(PKL_TRAIN_VERB_ANALYSIS, "wb") as f:
        pkl.dump(train_verb_counts, f)
        
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


"""
print("Noun Counts:")
i = 0
train_noun_counts = {noun: count for noun, count in train_noun_counts.items() if count > 1}
for noun, count in train_noun_counts.items():
    print(f"{noun}: {count}")
    i += 1
    if i >= 20:
        break
"""
i = 0
#print("\nNoun-Verb Counts:")
noun_verb_counts = {
    noun: {verb: count for verb, count in verb_counter.items() if count >= 5}
    for noun, verb_counter in train_verb_counts.items()
}

#for noun, verb_counter in noun_verb_counts.items():
#    print(f"{noun}: {verb_counter}")
#    i += 1
#    if i >= 5:
#        break

total_nouns = sum(train_noun_counts.values())
noun_percentages = {noun: (count / total_nouns) * 100 for noun, count in train_noun_counts.items()}

#print(f"Most Common: {val_noun_counts.most_common(200)}")
#print(f"Less Common: {val_noun_counts.most_common()[:-21:-1]}")


def get_less_common_labels(validation_nouns, training_nouns):
    total_val_nouns = sum(validation_nouns.values())
    val_percentages = {noun: (count / 200000) * 100 for noun, count in validation_nouns.items()}
    
    total_train_nouns = sum(training_nouns.values())
    train_percentages = {noun: (count / validation_nouns[noun]) * 100 for noun, count in training_nouns.items() if validation_nouns[noun]}
   
    less_common_labels = {}
    for noun, percentage in train_percentages.items():
        if noun in validation_nouns.keys():
            if percentage <= 45 and validation_nouns[noun] > 500: # More than 0.1% of images with that noun
                less_common_labels.update({noun: percentage})
        else:
            continue
       
    # print(less_common_labels)     
    return less_common_labels
    
less_common = get_less_common_labels(validation_nouns=val_noun_counts, training_nouns=train_noun_counts)

with open("sentences_info_more_500.txt", "w") as file:
    file.write("Let's do more sentences, I will give you the keyword noun and its most used verbs and the number of sentences for each one.\n\n")
    
    balancing_dict = {}
    for noun, _ in less_common.items():
        balancing_dict.update({noun: {}})
        number_of_sentences = int(val_noun_counts[noun] * 0.5) # We use 50% of the validation data
        for t_verb, t_freq in train_verb_counts[noun].items():
            if t_verb in val_verb_counts[noun].keys():
                index = val_verb_counts[noun][t_verb] - t_freq
                if index > 10:
                    balancing_dict[noun].update({t_verb : index})
 
        file.write(f"For the keyword '{noun}' do {number_of_sentences} different sentences please in JSON format.\n\n")
        file.write("These are the verbs used with this keyword and their weights:\n")
        file.write(str(balancing_dict[noun]) + "\n\n")
        
    
"""
print("Let's do more sentences, I will give you the keyword noun and its most used verbs and the number of sentences for each one.")
balancing_dict = {}
for noun, _ in less_common.items():
    balancing_dict.update({noun: {}})
    number_of_sentences = int(val_noun_counts[noun] * 0.5) #We do the 50% because we have half the validation data
    for t_verb, t_freq in train_verb_counts[noun].items():
        if t_verb in val_verb_counts[noun].keys():
            balancing_dict[noun].update({t_verb : val_verb_counts[noun][t_verb] - total_nouns})
    print(f"For the keyword '{noun}' do {number_of_sentences} different sentences please in json format.")
    print('')
    print('These are the verbs used with this keyword and each weights:')
    print(balancing_dict[noun])
    print('')
    break
"""
#print(less_common)
#for noun, percentage in noun_percentages.items():
#    print(f'{noun}: {percentage:.2f}%')
"""
min_counts = 20
less_common_train = {
    token: count
    for token, count in train_noun_counts.items()
    if count < val_noun_counts[token] * 0.5 and val_noun_counts[token] > min_counts
}
print(less_common_train)
print("Tokens less common in training compared to validation:")
for token, count in less_common_train.items():
    print(f"{token}: Training: {count} - Validation:  {val_noun_counts[token]}")
"""