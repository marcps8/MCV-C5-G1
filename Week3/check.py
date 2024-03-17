import pickle
import json
import os

ANNOTATIONS_JSON = (
    "/ghome/group01/MCV-C5-G1/Week3/inverted_annotations/inverted_annotations.json"
)
# Open the pickle file for reading
file_path = '/ghome/group01/MCV-C5-G1/Week3/pickles/task_e/dataloader_selective.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

if os.path.exists(ANNOTATIONS_JSON):
    with open(ANNOTATIONS_JSON, "r") as f:
        annotations = json.load(f)
        train_labels = annotations["train"]
        database_labels = annotations["database"]
        test_labels = annotations["test"]
        val_labels = annotations["val"]

# Display the first three examples
for i, example in enumerate(data[:3]):
    print(f"Example {i+1}: {example}")
    print(f"Anchor Labels: {train_labels[example[0]]}")
    print(f"Positive Labels: {train_labels[example[1]]}")
    print(f"Negative Labels: {train_labels[example[2]]}")