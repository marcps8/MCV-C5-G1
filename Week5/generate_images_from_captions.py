import json
from image_generator import generate_image

JSON_FILE = "new_captions_dict.json"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

def generate_images_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for item in data.values():
        caption = item["caption"]
        output_folder_positive = f"./generated_images/positives/"
        generate_image(prompt=caption, model_id=MODEL, output_folder=output_folder_positive)
        
        negative_caption = data[str(item['negative_id'])]['caption']
        output_folder_negative = f"./generated_images/negatives/"
        generate_image(prompt=negative_caption, model_id=MODEL, output_folder=output_folder_negative)

if __name__ == "__main__":
    generate_images_from_json(JSON_FILE)
