import matplotlib.pyplot as plt
import numpy as np

# Standard Strategy
first_global_accuracy = 0.5210
first_age_distribution = [0.2000, 0.0625, 0.4390, 0.5874, 0.5459, 0.3750, 0.0000]
first_gender_distribution = [0.5503, 0.4958]
first_ethnicity_distribution = [0.5281, 0.5263, 0.4769]
first_age_bias = 0.2827
first_gender_bias = 0.0546
first_ethnicity_bias = 0.0342


# Combined Strategy
second_global_accuracy = 0.5316
second_age_distribution = [0.0000, 0.0625, 0.4488, 0.6121, 0.5306, 0.3906, 0.0000]
second_gender_distribution = [0.5646, 0.5033]
second_ethnicity_distribution = [0.4944, 0.5449, 0.4444]
second_age_bias = 0.3127
second_gender_bias = 0.0613
second_ethnicity_bias = 0.0670

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
plt.legend()
plt.ylim(0, 0.5)
plt.tight_layout()
plt.savefig('bias_analysis_comparison.png')
plt.show()
