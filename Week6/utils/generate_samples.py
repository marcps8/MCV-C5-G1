import os
import torch
from diffusers import DiffusionPipeline

BASE_PROMPT      = "Human realistic single face of ethnicity {}, of age {} and gender {}."
OUTPUT_FOLDER    = "data/train_augmented/{}/"
GENDER_LABELS    = ['Male', 'Female']
ETHNICITY_LABELS = ['Asian', 'Caucasian', 'African-American']
AGE_RANGE_LABELS = ['10', '16', '22', '27', '40', '53', '65']
AGE_RANGE_DICT   = {'10': 1, '16': 2, '22': 3, '27': 4, '40': 5, '53': 6, '65': 7}

TUPLES_FEMALE_AFRICAN_AMERICAN = [
    (GENDER_LABELS[1], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[0], 150),
    (GENDER_LABELS[1], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[1], 150),
    (GENDER_LABELS[1], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[2], 120),
    (GENDER_LABELS[1], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[5], 130),
    (GENDER_LABELS[1], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[6], 150),    
]

TUPLES_MALE_AFRICAN_AMERICAN = [
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[0], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[1], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[2], 100),
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[4], 60),
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[5], 110),    
    (GENDER_LABELS[0], ETHNICITY_LABELS[2], AGE_RANGE_LABELS[6], 120),    
]

TUPLES_FEMALE_ASIAN = [
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[0], 120),
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[1], 120),
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[2], 100),
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[3], 40),
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[4], 100),
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[5], 120),    
    (GENDER_LABELS[1], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[6], 120),    
]

TUPLES_MALE_ASIAN = [
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[0], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[1], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[2], 100),
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[3], 100),
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[4], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[5], 120),    
    (GENDER_LABELS[0], ETHNICITY_LABELS[0], AGE_RANGE_LABELS[6], 120),    
]

TUPLES_FEMALE_CAUCASIAN = [
    (GENDER_LABELS[1], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[0], 120),
    (GENDER_LABELS[1], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[1], 80),
    (GENDER_LABELS[1], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[6], 100),    
]

TUPLES_MALE_CAUCASIAN = [
    (GENDER_LABELS[0], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[0], 120),
    (GENDER_LABELS[0], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[1], 20),
    (GENDER_LABELS[0], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[5], 30),    
    (GENDER_LABELS[0], ETHNICITY_LABELS[1], AGE_RANGE_LABELS[6], 90),    
]


def main():
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipeline.to("cuda")

    for gender, ethnicity, age_range, amount in TUPLES_MALE_CAUCASIAN:
        for id in range(amount):
            output_image = pipeline(
                prompt=BASE_PROMPT.format(ethnicity, age_range, gender)
            ).images[0]

            image_path = os.path.join(
                OUTPUT_FOLDER.format(AGE_RANGE_DICT[age_range]), 
                f"{gender}_{ethnicity}_{age_range}_{id}.png"
            )
            output_image.save(image_path)
    
if __name__ == "__main__":
    main()

