import json
from image_generator import generate_image

JSON_FILE = "new_captions.json"

def capts_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if i > 0 and i < 40:
            print(item["caption"],'\n')

if __name__ == "__main__":
    capts_from_json(JSON_FILE)