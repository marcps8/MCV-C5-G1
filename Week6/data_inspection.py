import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directory if it doesn't exist
plots_dir = './plots/train_downsampled'

# Load data
train_data = pd.read_csv('./data/train_set_age_labels.csv')
train_augmented_data = pd.read_csv('./data/train_augmented_set_age_labels.csv')
train_downsampled_data = pd.read_csv('./data/train_downsampled_set_age_labels.csv')
concat_train_data = pd.concat([train_data, train_augmented_data], ignore_index=True)
test_data = pd.read_csv('./data/test_set_age_labels.csv')
valid_data = pd.read_csv('./data/valid_set_age_labels.csv')

used_data = train_downsampled_data

# Map numerical codes to actual labels
gender_labels = {1: 'Male', 2: 'Female'}
ethnicity_labels = {1: 'Asian', 2: 'Caucasian', 3: 'African-American'}
age_range_labels = {
    1: '[7-13] years',
    2: '[14-18] years',
    3: '[19-24] years',
    4: '[25-32] years',
    5: '[33-45] years',
    6: '[46-60] years',
    7: '61+ years'
}

# Define colors for bar charts
bar_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

#################
## BAR CHARTS  ##
#################

grouped = used_data.groupby(['AgeGroup', 'Gender']).size().unstack(fill_value=0)
grouped.columns = [gender_labels[x] for x in grouped.columns]
grouped.index = [age_range_labels[x] for x in grouped.index]

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
grouped.plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Frequency of Gender Across Age Ranges')
plt.xlabel('Age Range')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'age_distribution_gender.png'))
plt.close()

# Plot age distribution for each ethnicity category
grouped = used_data.groupby(['AgeGroup', 'Ethnicity']).size().unstack(fill_value=0)
grouped.columns = [ethnicity_labels[x] for x in grouped.columns]
grouped.index = [age_range_labels[x] for x in grouped.index]

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
grouped.plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Frequency of Ethnicity Across Age Ranges')
plt.xlabel('Age Range')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'age_distribution_ethnicity.png'))
plt.close()


used_data['Gender_Ethnicity'] = used_data.apply(
    lambda row: f"{gender_labels[row['Gender']]}, {ethnicity_labels[row['Ethnicity']]}", axis=1
)
grouped = used_data.groupby(['AgeGroup', 'Gender_Ethnicity']).size().unstack(fill_value=0)
grouped.index = [age_range_labels[x] for x in grouped.index]

# Create bar plot
fig, ax = plt.subplots(figsize=(12, 8))
grouped.plot(kind='bar', ax=ax, colormap='viridis')  # Using a colormap to differentiate the combinations
plt.title('Frequency of Gender and Ethnicity Combinations Across Age Ranges')
plt.xlabel('Age Range')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Gender, Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(plots_dir, 'age_distribution_gender_ethnicity.png'))
plt.show()


#################
## PIE CHARTS  ##
#################

# Plot age distribution for each gender category in pie charts
for gender_code, gender_label in gender_labels.items():
    plt.figure(figsize=(8, 8))
    gender_data = used_data[used_data['Gender'] == gender_code]
    age_counts = gender_data['AgeGroup'].map(age_range_labels).value_counts().sort_index()
    plt.pie(age_counts, labels=None, autopct=None, startangle=140, colors=bar_colors[:len(age_counts)])
    plt.title(f'Age Distribution for {gender_label}')
    plt.axis('equal')
    plt.legend([f"{label} ({age_counts[i]} [{100 * age_counts[i] / len(gender_data):.1f}%])" for i, label in enumerate(age_counts.index)], loc="best", fontsize='small')
    plt.savefig(os.path.join(plots_dir, f'age_distribution_gender_{gender_label.lower()}.png'))
    plt.close()

# Plot age distribution for each ethnicity category in pie charts
for ethnicity_code, ethnicity_label in ethnicity_labels.items():
    plt.figure(figsize=(8, 8))
    ethnicity_data = used_data[used_data['Ethnicity'] == ethnicity_code]
    age_counts = ethnicity_data['AgeGroup'].map(age_range_labels).value_counts().sort_index()
    plt.pie(age_counts, labels=None, autopct=None, startangle=140, colors=bar_colors[:len(age_counts)])
    plt.title(f'Age Distribution for {ethnicity_label}')
    plt.axis('equal')
    plt.legend([f"{label} ({age_counts[i]} [{100 * age_counts[i] / len(ethnicity_data):.1f}%])" for i, label in enumerate(age_counts.index)], loc="best", fontsize='small')
    plt.savefig(os.path.join(plots_dir, f'age_distribution_ethnicity_{ethnicity_label.lower()}.png'))
    plt.close()

# Plot age distribution for each gender category, for each ethnicity category in pie charts
for gender_code, gender_label in gender_labels.items():
    for ethnicity_code, ethnicity_label in ethnicity_labels.items():
        plt.figure(figsize=(8, 8))
        subset = used_data[(used_data['Gender'] == gender_code) & (used_data['Ethnicity'] == ethnicity_code)]
        label = f'{gender_label} - {ethnicity_label}'
        age_counts = subset['AgeGroup'].map(age_range_labels).value_counts().sort_index()
        plt.pie(age_counts, labels=None, autopct=None, startangle=140, colors=bar_colors[:len(age_counts)])
        plt.title(f'Age Distribution for {label}')
        plt.axis('equal')
        plt.legend([f"{label} ({age_counts[i]} [{100 * age_counts[i] / len(subset):.1f}%])" for i, label in enumerate(age_counts.index)], loc="best", fontsize='small')
        plt.savefig(os.path.join(plots_dir, f'age_distribution_gender_ethnicity_{gender_label.lower()}_{ethnicity_label.lower()}.png'))
        plt.close()
