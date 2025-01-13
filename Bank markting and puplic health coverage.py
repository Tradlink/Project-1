import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, zscore

# Load the datasets
bank_train = pd.read_csv('path/to/bank_marketing_training.csv')
ASEC_df = pd.read_csv('path/to/ASEC.csv')

# Data preprocessing and initial setup
bank_train['index'] = pd.Series(range(bank_train.shape[0]))
bank_train['days_since_previous'] = bank_train['days_since_previous'].replace({999: np.NaN})

# Plot histogram of days since previous contact
bank_train['days_since_previous'].plot(kind='hist', edgecolor='black', color='blue')
plt.xlabel('Number of Days')
plt.title('Histogram of Days Since Previous Contact')
plt.show()

# Boxplot for days since previous contact
bank_train['days_since_previous'].plot(kind='box')
plt.ylabel('Number of Days')
plt.title('Boxplot of Days Since Previous Contact')
plt.show()

# Calculate IQR and detect outliers
Q1 = np.nanpercentile(bank_train['days_since_previous'], 25)
Q3 = np.nanpercentile(bank_train['days_since_previous'], 75)
IQR = Q3 - Q1
upfence = Q3 + 1.5 * IQR
outliers = bank_train.query('days_since_previous > @upfence')

# Visualizing education distribution
ed_counts = {
    "basic.4y": 2688, "basic.6yr": 1498, "basic.9yr": 4050,
    "high.school": 6130, "illiterate": 12, "professional.course": 3423,
    "university.degree": 7946, "unknown": 1127
}
categories, counts = list(ed_counts.keys()), list(ed_counts.values())
plt.barh(categories, counts, color='skyblue', edgecolor='black')
plt.xlabel('Count')
plt.title('Distribution of Education Level')
plt.show()

# Education levels mapped to numeric values
edu_mapping = {
    'illiterate': 0, 'basic.4y': 4, 'basic.6y': 6,
    'basic.9y': 9, 'high.school': 12,
    'professional.course': 12, 'university.degree': 16, 'unknown': np.NaN
}
bank_train['education_numeric'] = bank_train['education'].map(edu_mapping)

# Plot normalized education distribution
cleaned_bank = bank_train.dropna(subset=['education_numeric'])
edu_counts = cleaned_bank['education_numeric'].value_counts(sort=False).sort_index()
plt.barh(edu_counts.index, edu_counts.values, color='salmon', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Education Level (Years)')
plt.title('Distribution of Education Level (Cleaned Data)')
plt.show()

# Age analysis
bin_width = 5
plt.hist(ASEC_df['A_AGE'], bins=range(0, 86, bin_width), edgecolor='black', color='green')
plt.xlabel('Age')
plt.title('Histogram of Age')
plt.show()

# Public health coverage analysis
ASEC_df['NOW_PUB'] = ASEC_df['NOW_PUB'].replace({1: 'Yes', 2: 'No'})

