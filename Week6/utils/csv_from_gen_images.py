import os
import pandas as pd

DATA_DIR         = 'data/'
TRAIN_DIR        = DATA_DIR + "train/"

GENDER_DICT      = {'Male': 1, 'Female': 2}
ETHNICITY_DICT   = {'Asian': 1, 'Caucasian': 2, 'African-American': 3}
AGE_RANGE_DICT   = {'10': 1, '16': 2, '22': 3, '27': 4, '40': 5, '53': 6, '65': 7}

def main():
    data = {
        "VideoName": [],
        "UserID": [],
        "AgeGroup": [],
        "Gender": [],
        "Ethnicity": []
    }
    index = 0
    for folder in os.listdir(TRAIN_DIR):
        curr_image_dir = TRAIN_DIR + folder
        for image in os.listdir(curr_image_dir):
            splitted_name = image.split("_")
            gender, ethnicity, age_group = splitted_name[0], splitted_name[1], splitted_name[2]
            data["VideoName"].append(image)
            data["UserID"].append(index)
            data["AgeGroup"].append(AGE_RANGE_DICT[age_group])
            data["Gender"].append(GENDER_DICT[gender])
            data["Ethnicity"].append(ETHNICITY_DICT[ethnicity])
            index += 1
    
    csv_name = DATA_DIR + "train_undersampled_set_age_labels.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_name, index=False)

if __name__ == "__main__":
    main()