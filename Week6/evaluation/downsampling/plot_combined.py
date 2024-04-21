import matplotlib.pyplot as plt
import numpy as np


# Standard Strategy
first_global_accuracy = 0.4511
first_age_distribution = [0.8000, 0.3542, 0.4073, 0.4712, 0.4563, 0.4375, 0.4444]
first_gender_distribution = [0.4705, 0.4345]
first_ethnicity_distribution = [0.4719, 0.4449, 0.4907]
first_age_bias = 0.1413
first_gender_bias = 0.0360
first_ethnicity_bias = 0.0306


# Combined Strategy
second_global_accuracy = 0.4425
second_age_distribution = [0.8000, 0.3542, 0.3780, 0.4712, 0.4541, 0.4062, 0.3333]
second_gender_distribution = [0.4639, 0.4241]
second_ethnicity_distribution = [0.4719, 0.4311, 0.5185]
second_age_bias = 0.1629
second_gender_bias = 0.0398
second_ethnicity_bias = 0.0583

# Age Distribution Comparison
age_categories = ['[7-13] years', '[14-18] years', '[19-24] years', '[25-32] years', '[33-45] years', '[46-60] years', '61+ years']
bar_width = 0.35
index = np.arange(len(age_categories))

plt.figure(figsize=(10, 6))
plt.bar(index - bar_width/2, first_age_distribution, bar_width, label='Standard Strategy')
plt.bar(index + bar_width/2, second_age_distribution, bar_width, label='Combined Strategy')
plt.xlabel('Age Categories')
plt.ylabel('Accuracy')
plt.title('Age Distribution Comparison')
plt.xticks(index, age_categories)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('age_distribution_comparison.png')
plt.show()

# Gender Distribution Comparison
gender_categories = ['Male', 'Female']

plt.figure(figsize=(6, 6))
plt.bar(np.arange(len(gender_categories)), first_gender_distribution, bar_width, label='Standard Strategy')
plt.bar(np.arange(len(gender_categories)) + bar_width, second_gender_distribution, bar_width, label='Combined Strategy')
plt.xlabel('Gender')
plt.ylabel('Accuracy')
plt.title('Gender Distribution Comparison')
plt.xticks(np.arange(len(gender_categories)) + bar_width/2, gender_categories)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('gender_distribution_comparison.png')
plt.show()

# Ethnicity Distribution Comparison
ethnicity_categories = ['Asian', 'Caucasian', 'African-American']

plt.figure(figsize=(8, 6))
plt.bar(np.arange(len(ethnicity_categories)), first_ethnicity_distribution, bar_width, label='Standard Strategy')
plt.bar(np.arange(len(ethnicity_categories)) + bar_width, second_ethnicity_distribution, bar_width, label='Combined Strategy')
plt.xlabel('Ethnicity')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Ethnicity Distribution Comparison')
plt.xticks(np.arange(len(ethnicity_categories)) + bar_width/2, ethnicity_categories)
plt.legend()
plt.tight_layout()
plt.savefig('ethnicity_distribution_comparison.png')
plt.show()

# Bias Analysis Comparison
bias_labels = ['Age Bias', 'Gender Bias', 'Ethnicity Bias']
first_bias_values = [first_age_bias, first_gender_bias, first_ethnicity_bias]
second_bias_values = [second_age_bias, second_gender_bias, second_ethnicity_bias]

plt.figure(figsize=(8, 6))
plt.bar(bias_labels, first_bias_values, bar_width, label='Standard Strategy')
plt.bar(np.arange(len(bias_labels)) + bar_width, second_bias_values, bar_width, label='Combined Strategy')
plt.xlabel('Bias')
plt.ylabel('Bias Value')
plt.title('Bias Analysis Comparison')
plt.ylim(0, 0.5)
plt.legend()
plt.tight_layout()
plt.savefig('bias_analysis_comparison.png')
plt.show()
