import matplotlib.pyplot as plt
import numpy as np

# Standard Strategy
first_global_accuracy = 0.5559
first_age_distribution = [0.6000, 0.0000, 0.3512, 0.7603, 0.4301, 0.1250, 0.3889]
first_gender_distribution = [0.5536, 0.5580]
first_ethnicity_distribution = [0.4607, 0.5617, 0.5509]
first_age_bias = 0.3152
first_gender_bias = 0.0044
first_ethnicity_bias = 0.0673

# Combined Strategy
second_global_accuracy = 0.5782
second_age_distribution = [0.8000, 0.0000, 0.3683, 0.7901, 0.4410, 0.1719, 0.3333]
second_gender_distribution = [0.5777, 0.5787]
second_ethnicity_distribution = [0.4157, 0.5844, 0.5972]
second_age_bias = 0.3566
second_gender_bias = 0.0010
second_ethnicity_bias = 0.1210

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
