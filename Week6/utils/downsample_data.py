import os
import shutil
import pandas as pd

DATA_DIR  = 'data/'

TRAIN_DATA           = pd.read_csv('data/train_set_age_labels.csv')
TRAIN_AUGMENTED_DATA = pd.read_csv('data/train_augmented_set_age_labels.csv')
CONCAT_TRAIN_DATA    = pd.concat([TRAIN_DATA, TRAIN_AUGMENTED_DATA], ignore_index=True)

def check_groupped(df):
    grouped = df.groupby(['AgeGroup', 'Gender', 'Ethnicity'])
    group_counts = grouped.size().reset_index(name='Count')
    print(group_counts)

def downsample_csv():
    grouped = CONCAT_TRAIN_DATA.groupby(['AgeGroup', 'Gender', 'Ethnicity'])
    group_counts = grouped.size().reset_index(name='Count')
    min_count = group_counts['Count'].min()
    
    def downsampled(group):
        return group.sample(n=min_count)

    balanced_df = grouped.apply(downsampled).reset_index(drop=True)
    csv_name = DATA_DIR + "train_downsampledd_set_age_labels.csv"
    balanced_df.to_csv(csv_name, index=False)

def downsample_folder():    
    src_train_dir = DATA_DIR + "train_combined/"
    target_train_dir = DATA_DIR + "train_downsampledd"
    
    csv_name = DATA_DIR + "train_downsampledd_set_age_labels.csv"
    df = pd.read_csv(csv_name)
    for video_name, _, age_group, _, _ in df.values:
        video_name = video_name.replace("mp4", "jpg")
        src = os.path.join(src_train_dir, str(age_group), video_name)
        target = os.path.join(target_train_dir, str(age_group), video_name)
        shutil.copy(src, target)
        
if __name__ == "__main__":
    #downsample_csv()
    downsample_folder() 
