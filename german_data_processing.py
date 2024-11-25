# data_processing.py

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from sklearn.model_selection import train_test_split


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

# Load the dataset
data = pd.read_csv('german.data', sep=' ', header=None, names=column_names)

# Initialize a dictionary to store label encoders for categorical variables
label_encoders = {}

# List of categorical columns for encoding
categorical_cols = [
    'Status_of_existing_checking_account', 'Credit_history', 'Purpose',
    'Savings_account_bonds', 'Present_employment_since',
    'Personal_status_and_sex', 'Other_debtors_guarantors', 'Property',
    'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker'
]

# Apply label encoding to categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save the encoder for future use

# Save the encoders to a file for reuse
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Map target variable: 1 (Good) to 0, 2 (Bad) to 1
data['Target'] = data['Target'].map({1: 0, 2: 1})

# List of numerical columns for standardization
numerical_cols = [
    'Duration_in_month', 'Credit_amount',
    'Installment_rate_in_percentage_of_disposable_income',
    'Present_residence_since', 'Age_in_years',
    'Number_of_existing_credits_at_this_bank',
    'Number_of_people_being_liable_to_provide_maintenance_for'
]

# Initialize and apply standard scaling to numerical columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save the scaler to a file for reuse
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the processed dataset to a CSV file
data.to_csv('german_credit_processed.csv', index=False)

# Split the processed dataset into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(
    data, test_size=0.2, stratify=data['Target'], random_state=42
)

# Save the training and testing sets to separate CSV files
train_data.to_csv('german_credit_train.csv', index=False)
test_data.to_csv('german_credit_test.csv', index=False)

# Output a message indicating successful splitting
print("Data splitting complete. Training data saved as 'german_credit_train.csv'.")
print("Testing data saved as 'german_credit_test.csv'.")


# Output a message indicating successful completion
print("Data processing complete. Processed data saved as 'german_credit_processed.csv'.")
print("Encoders and scaler saved as 'label_encoders.pkl' and 'scaler.pkl'.")
