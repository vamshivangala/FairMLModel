# data_processing_with_encodings_and_saving.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from sklearn.model_selection import train_test_split

# Define column names as per the dataset description
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Load the dataset
# Ensure you have 'adult.data' in your working directory or provide the correct path
data = pd.read_csv('adult.data', header=None, names=column_names, na_values=' ?')

# Handle missing values by dropping rows with missing values
data.dropna(inplace=True)

# Dictionary to store encodings
encodings = {}

# Encode categorical variables
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    encodings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Print and save the encodings
print("Encodings for categorical variables:")
for col, mapping in encodings.items():
    print(f"{col}: {mapping}")

# Save the encodings to a file for future reference
with open('adult_categorical_encodings.pkl', 'wb') as f:
    pickle.dump(encodings, f)

# Encode the target variable
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Create 'age_group' as a protected attribute (e.g., under 40 and 40 or over)
data['age_group'] = data['age'].apply(lambda x: 0 if x < data['age'].median() else 1)

# Optional: Feature scaling for numerical columns
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save the scaler to a file for future use
with open('adult_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the processed data to a CSV file
data.to_csv('adult_data_processed.csv', index=False)

train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['income'], random_state=42)
train_data.to_csv('adult_train_data.csv', index=False)
test_data.to_csv('adult_test_data.csv', index=False)


print("Data preprocessing completed and saved to 'adult_data_processed.csv'.")
print("Encodings saved to 'adult_categorical_encodings.pkl'.")
