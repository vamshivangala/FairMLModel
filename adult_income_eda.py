# adult_income_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Define column names as per the dataset description
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Load the original data (before preprocessing)
original_data = pd.read_csv(
    'adult.data',
    header=None,
    names=column_names,
    skipinitialspace=True,
    na_values='?'
)
original_data.dropna(inplace=True)

# Load the preprocessed data
data = pd.read_csv('adult_data_processed.csv')

# Replace scaled 'age' and 'hours-per-week' with original values
data['age'] = original_data['age'].values
data['hours-per-week'] = original_data['hours-per-week'].values

# List of categorical columns that were encoded
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]

# Recreate label encoders and inverse transform the encoded values
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(original_data[col])
    label_encoders[col] = le
    # Inverse transform the encoded values
    data[col] = le.inverse_transform(data[col])

# Map the target variable back to original categories
data['income'] = data['income'].map({0: '<=50K', 1: '>50K'})

# Map 'age_group' back to categorical labels if applicable
if 'age_group' in data.columns:
    data['age_group'] = data['age_group'].map({0: 'Under Median Age', 1: 'Over Median Age'})

# Define the target variable and protected attributes
target_variable = 'income'
protected_attributes = ['race', 'sex', 'age_group']

# Exclude 'education_num' from features
features = data.columns.tolist()
features = [feat for feat in features if feat not in protected_attributes + [target_variable, 'education_num']]

# ----------------------------------------
# 1. Single-Variable Analysis
# ----------------------------------------

print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Histograms for numerical variables
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Exclude 'education_num', 'age', 'hours-per-week' (handled separately), and protected attributes
numerical_columns = [col for col in numerical_columns if col not in protected_attributes + [target_variable, 'education_num', 'age', 'hours-per-week']]

# Bin 'age' into groups of 5 years
data['age_grouped'] = pd.cut(data['age'], bins=range(0, 101, 5), right=False)

# Sort the age groups based on the left endpoint of the intervals
age_group_order = sorted(data['age_grouped'].dropna().unique(), key=lambda x: x.left)

# Plot distribution of 'age_grouped'
plt.figure(figsize=(12, 6))
sns.countplot(x='age_grouped', data=data, order=age_group_order)
plt.title('Distribution of Age Groups (5-year bins)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot distributions for 'age' and 'hours-per-week' with actual values
for col in ['age', 'hours-per-week']:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col.replace("-", " ").capitalize()}')
    plt.xlabel(col.replace('-', ' ').capitalize())
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot distributions for other numerical variables
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col.replace("_", " ").capitalize()}')
    plt.xlabel(col.replace('_', ' ').capitalize())
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Bar plots for categorical variables
categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
# Exclude target variable, 'age_grouped' and protected attributes from categorical columns
categorical_columns = [col for col in categorical_columns if col not in [target_variable, 'age_grouped'] + protected_attributes]

for col in categorical_columns + protected_attributes + [target_variable]:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=data, order=data[col].value_counts().index)
    plt.title(f'Count of {col.replace("_", " ").capitalize()}')
    plt.xlabel(col.replace("_", " ").capitalize())
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# 2. Two-Factor Analysis
# ----------------------------------------

# Correlation Matrix for numerical variables
corr_columns = ['age', 'hours-per-week'] + numerical_columns
corr_matrix = data[corr_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Relationship between Protected Attributes and Target Variable
for attr in protected_attributes:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=attr, hue=target_variable, data=data)
    plt.title(f'{target_variable.capitalize()} by {attr.replace("_", " ").capitalize()}')
    plt.xlabel(attr.replace("_", " ").capitalize())
    plt.ylabel('Count')
    plt.legend(title='Income')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Relationship between 'age_grouped' and Target Variable
plt.figure(figsize=(12, 6))
sns.countplot(x='age_grouped', hue=target_variable, data=data, order=age_group_order)
plt.title(f'{target_variable.capitalize()} by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Income')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Relationship between Protected Attributes and Other Features
for attr in protected_attributes:
    for col in numerical_columns + ['age', 'hours-per-week']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=attr, y=col, data=data)
        plt.title(f'{col.replace("_", " ").capitalize()} by {attr.replace("_", " ").capitalize()}')
        plt.xlabel(attr.replace("_", " ").capitalize())
        plt.ylabel(col.replace("_", " ").capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ----------------------------------------
# Additional Income-Related Plots
# ----------------------------------------

# Plot Income by Education Level
plt.figure(figsize=(12, 6))
sns.countplot(x='education', hue='income', data=data, order=sorted(data['education'].unique()))
plt.title('Income by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()
plt.show()

# Plot Income by Occupation
plt.figure(figsize=(12, 6))
sns.countplot(x='occupation', hue='income', data=data, order=sorted(data['occupation'].unique()))
plt.title('Income by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()
plt.show()

# Plot Income by Work Hours per Week
plt.figure(figsize=(12, 6))
sns.boxplot(x='income', y='hours-per-week', data=data)
plt.title('Income by Work Hours per Week')
plt.xlabel('Income')
plt.ylabel('Hours per Week')
plt.tight_layout()
plt.show()

# Additional Income vs. Other Variables Plots

# Example: Income by Marital Status
plt.figure(figsize=(12, 6))
sns.countplot(x='marital-status', hue='income', data=data, order=sorted(data['marital-status'].unique()))
plt.title('Income by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()
plt.show()

# Example: Income by Race
plt.figure(figsize=(12, 6))
sns.countplot(x='race', hue='income', data=data, order=sorted(data['race'].unique()))
plt.title('Income by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()
plt.show()


# ----------------------------------------
# Other Graphs (Commented Out)
# ----------------------------------------

# Plot Income by Occupation
# plt.figure(figsize=(12, 6))
# sns.countplot(x='occupation', hue='income', data=data, order=sorted(data['occupation'].unique()))
# plt.title('Income by Occupation')
# plt.xlabel('Occupation')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.legend(title='Income')
# plt.tight_layout()
# plt.show()

# # Plot Income by Work Hours per Week
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='income', y='hours-per-week', data=data)
# plt.title('Income by Work Hours per Week')
# plt.xlabel('Income')
# plt.ylabel('Hours per Week')
# plt.tight_layout()
# plt.show()

# # Example: Income by Marital Status
# plt.figure(figsize=(12, 6))
# sns.countplot(x='marital-status', hue='income', data=data, order=sorted(data['marital-status'].unique()))
# plt.title('Income by Marital Status')
# plt.xlabel('Marital Status')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.legend(title='Income')
# plt.tight_layout()
# plt.show()

# # Example: Income by Race
# plt.figure(figsize=(12, 6))
# sns.countplot(x='race', hue='income', data=data, order=sorted(data['race'].unique()))
# plt.title('Income by Race')
# plt.xlabel('Race')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.legend(title='Income')
# plt.tight_layout()
# plt.show()