# german_credit_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------
# Load Data
# ----------------------------------------

# Define column names based on the dataset description
column_names = [
    'Status_of_existing_checking_account', 'Duration_in_month',
    'Credit_history', 'Purpose', 'Credit_amount', 'Savings_account_bonds',
    'Present_employment_since', 'Installment_rate_in_percentage_of_disposable_income',
    'Personal_status_and_sex', 'Other_debtors_guarantors',
    'Present_residence_since', 'Property', 'Age_in_years',
    'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for',
    'Telephone', 'Foreign_worker', 'Target'
]

# Load the original data (before preprocessing)
original_data = pd.read_csv('german.data', sep=' ', header=None, names=column_names)

# Load the preprocessed data
processed_data = pd.read_csv('german_credit_processed.csv')

# Replace scaled numerical variables with original values
numerical_cols = [
    'Duration_in_month', 'Credit_amount',
    'Installment_rate_in_percentage_of_disposable_income',
    'Present_residence_since', 'Age_in_years',
    'Number_of_existing_credits_at_this_bank',
    'Number_of_people_being_liable_to_provide_maintenance_for'
]

for col in numerical_cols:
    processed_data[col] = original_data[col]

# Map encoded categorical variables back to original codes
categorical_cols = [
    'Status_of_existing_checking_account', 'Credit_history', 'Purpose',
    'Savings_account_bonds', 'Present_employment_since',
    'Personal_status_and_sex', 'Other_debtors_guarantors', 'Property',
    'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker'
]

# Recreate label encoders and inverse transform the encoded values
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(original_data[col])
    label_encoders[col] = le
    processed_data[col] = le.inverse_transform(processed_data[col])

# Map target variable back to original categories
processed_data['Target'] = processed_data['Target'].map({0: 'Good', 1: 'Bad'})

# ----------------------------------------
# Map Codes to Descriptive Labels
# ----------------------------------------

# Create mapping dictionaries based on the dataset description
# For brevity, I'll show mappings for a few attributes. You need to complete the mappings for all attributes as per your requirements.

status_checking_account_mapping = {
    'A11': '... < 0 DM',
    'A12': '0 <= ... < 200 DM',
    'A13': '... >= 200 DM / salary assignments for at least 1 year',
    'A14': 'no checking account'
}

credit_history_mapping = {
    'A30': 'no credits taken/ all credits paid back duly',
    'A31': 'all credits at this bank paid back duly',
    'A32': 'existing credits paid back duly till now',
    'A33': 'delay in paying off in the past',
    'A34': 'critical account/ other credits existing (not at this bank)'
}

purpose_mapping = {
    'A40': 'car (new)',
    'A41': 'car (used)',
    'A42': 'furniture/equipment',
    'A43': 'radio/television',
    'A44': 'domestic appliances',
    'A45': 'repairs',
    'A46': 'education',
    'A48': 'retraining',
    'A49': 'business',
    'A410': 'others'
}

savings_account_mapping = {
    'A61': '... < 100 DM',
    'A62': '100 <= ... < 500 DM',
    'A63': '500 <= ... < 1000 DM',
    'A64': '.. >= 1000 DM',
    'A65': 'unknown/ no savings account'
}

employment_since_mapping = {
    'A71': 'unemployed',
    'A72': '... < 1 year',
    'A73': '1 <= ... < 4 years',
    'A74': '4 <= ... < 7 years',
    'A75': '.. >= 7 years'
}

personal_status_sex_mapping = {
    'A91': 'male: divorced/separated',
    'A92': 'female: divorced/separated/married',
    'A93': 'male: single',
    'A94': 'male: married/widowed',
    'A95': 'female: single'
}

other_debtors_mapping = {
    'A101': 'none',
    'A102': 'co-applicant',
    'A103': 'guarantor'
}

# Continue mapping for other categorical variables...

# Apply mappings to the data
processed_data['Status_of_existing_checking_account'] = processed_data['Status_of_existing_checking_account'].map(status_checking_account_mapping)
processed_data['Credit_history'] = processed_data['Credit_history'].map(credit_history_mapping)
processed_data['Purpose'] = processed_data['Purpose'].map(purpose_mapping)
processed_data['Savings_account_bonds'] = processed_data['Savings_account_bonds'].map(savings_account_mapping)
processed_data['Present_employment_since'] = processed_data['Present_employment_since'].map(employment_since_mapping)
processed_data['Personal_status_and_sex'] = processed_data['Personal_status_and_sex'].map(personal_status_sex_mapping)
processed_data['Other_debtors_guarantors'] = processed_data['Other_debtors_guarantors'].map(other_debtors_mapping)
# Map other categorical variables similarly...

# ----------------------------------------
# EDA
# ----------------------------------------

data = processed_data.copy()

# List of features
features = data.columns.tolist()

# Identify the target variable and protected attributes
target_variable = 'Target'
protected_attributes = ['Personal_status_and_sex', 'Age_group']

# ----------------------------------------
# 1. Single-Variable Analysis
# ----------------------------------------

print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Histograms for numerical variables
numerical_columns = numerical_cols.copy()

# Bin 'Age_in_years' into 5-year intervals
data['Age_group'] = pd.cut(data['Age_in_years'], bins=range(15, 75, 5), right=False)
age_group_order = sorted(data['Age_group'].dropna().unique(), key=lambda x: x.left)

# Plot distribution of 'Age_group'
plt.figure(figsize=(12, 6))
sns.countplot(x='Age_group', data=data, order=age_group_order)
plt.title('Distribution of Age Groups (5-year bins)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot distributions for numerical variables
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col.replace("_", " ").capitalize()}')
    plt.xlabel(col.replace('_', ' ').capitalize())
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Bar plots for categorical variables
categorical_columns = [
    'Status_of_existing_checking_account', 'Credit_history', 'Purpose',
    'Savings_account_bonds', 'Present_employment_since',
    'Personal_status_and_sex', 'Other_debtors_guarantors', 'Property',
    'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker'
]

for col in categorical_columns + [target_variable]:
    plt.figure(figsize=(12, 6))
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

# Correlation Matrix
corr_columns = numerical_columns.copy()
corr_matrix = data[corr_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Correlation with Target Variable
print("\nCorrelation with Target Variable ('Target'):")
# Need to convert 'Target' to numerical for correlation
data['Target_numeric'] = data['Target'].map({'Good': 0, 'Bad': 1})
print(corr_matrix := data[corr_columns + ['Target_numeric']].corr()['Target_numeric'].sort_values(ascending=False))

# Relationship between Protected Attributes and Target Variable
for attr in protected_attributes:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=attr, hue=target_variable, data=data, order=data[attr].value_counts().index)
    plt.title(f'{target_variable} by {attr.replace("_", " ").capitalize()}')
    plt.xlabel(attr.replace("_", " ").capitalize())
    plt.ylabel('Count')
    plt.legend(title=target_variable)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Relationship between 'Age_group' and Target Variable
plt.figure(figsize=(12, 6))
sns.countplot(x='Age_group', hue=target_variable, data=data, order=age_group_order)
plt.title(f'{target_variable} by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title=target_variable)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Relationship between Protected Attributes and Other Features
for attr in protected_attributes:
    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=attr, y=col, data=data)
        plt.title(f'{col.replace("_", " ").capitalize()} by {attr.replace("_", " ").capitalize()}')
        plt.xlabel(attr.replace("_", " ").capitalize())
        plt.ylabel(col.replace("_", " ").capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ----------------------------------------
# Additional Plots: Savings and Employment Status by Credit Risk
# ----------------------------------------

# Credit Risk by Savings Account/Bonds
plt.figure(figsize=(12, 6))
sns.countplot(x='Savings_account_bonds', hue='Target', data=data)
plt.title('Credit Risk by Savings Account/Bonds')
plt.xlabel('Savings Account/Bonds')
plt.ylabel('Count')
plt.legend(title='Credit Risk')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Credit Risk by Employment Duration
plt.figure(figsize=(12, 6))
sns.countplot(x='Present_employment_since', hue='Target', data=data)
plt.title('Credit Risk by Employment Duration')
plt.xlabel('Employment Duration')
plt.ylabel('Count')
plt.legend(title='Credit Risk')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Credit Risk by Housing Status
plt.figure(figsize=(12, 6))
sns.countplot(x='Housing', hue='Target', data=data)
plt.title('Credit Risk by Housing Status')
plt.xlabel('Housing Status')
plt.ylabel('Count')
plt.legend(title='Credit Risk')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # 1. Savings Account by Credit Risk
# plt.figure(figsize=(12, 6))
# sns.countplot(x='Savings_account_bonds', hue='Target', data=data)
# plt.title('Credit Risk by Savings Account Status')
# plt.xlabel('Savings Account/Bonds')
# plt.ylabel('Count')
# plt.legend(title='Credit Risk')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 2. Employment Status by Credit Risk
# plt.figure(figsize=(12, 6))
# sns.countplot(x='Present_employment_since', hue='Target', data=data)
# plt.title('Credit Risk by Employment Status')
# plt.xlabel('Employment Status')
# plt.ylabel('Count')
# plt.legend(title='Credit Risk')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 3. Savings Account by Credit Risk Segmented by Gender
# plt.figure(figsize=(12, 6))
# sns.countplot(
#     x='Savings_account_bonds',
#     hue='Personal_status_and_sex',
#     data=data,
#     hue_order=[
#         'male: divorced/separated', 
#         'female: divorced/separated/married', 
#         'male: single', 
#         'male: married/widowed', 
#         'female: single'
#     ]
# )
# plt.title('Credit Risk by Savings Account Status Segmented by Gender')
# plt.xlabel('Savings Account/Bonds')
# plt.ylabel('Count')
# plt.legend(title='Personal Status and Sex')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 4. Employment Status by Credit Risk Segmented by Gender
# plt.figure(figsize=(12, 6))
# sns.countplot(
#     x='Present_employment_since',
#     hue='Personal_status_and_sex',
#     data=data,
#     hue_order=[
#         'male: divorced/separated', 
#         'female: divorced/separated/married', 
#         'male: single', 
#         'male: married/widowed', 
#         'female: single'
#     ]
# )
# plt.title('Credit Risk by Employment Status Segmented by Gender')
# plt.xlabel('Employment Status')
# plt.ylabel('Count')
# plt.legend(title='Personal Status and Sex')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()