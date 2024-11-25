import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pickle

# Load the preprocessed data
data = pd.read_csv('german_credit_processed.csv')

# Load the saved label encodings for reverse mapping
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Reverse map "Personal_status_and_sex" and "Target" for better readability
data['Personal_status_and_sex'] = data['Personal_status_and_sex'].map(
    {i: label_encoders['Personal_status_and_sex'].classes_[i] for i in range(len(label_encoders['Personal_status_and_sex'].classes_))}
)
data['Target'] = data['Target'].map({0: 'Good', 1: 'Bad'})

# Define a mapping for the "Personal_status_and_sex" variable
personal_status_mapping = {
    'A91': 'Male: Divorced/Separated',
    'A92': 'Female: Divorced/Separated/Married',
    'A93': 'Male: Single',
    'A94': 'Male: Married/Widowed',
    'A95': 'Female: Single'
}

# Apply the mapping to replace encoded values with descriptive labels
data['Personal_status_and_sex'] = data['Personal_status_and_sex'].replace(personal_status_mapping)

# ----------------------------------------
# Proportions Plot
# ----------------------------------------

# Compute proportions for "Personal Status and Sex" vs. "Target"
status_target = data.groupby(['Personal_status_and_sex', 'Target']).size().unstack(fill_value=0)
status_target_ratio = status_target.div(status_target.sum(axis=1), axis=0)  # Normalize rows

# Plot stacked bar chart
colors = ['lightcoral', 'lightblue']
status_target_ratio.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), edgecolor='black')
plt.title('Credit Risk Distribution by Personal Status and Sex')
plt.xlabel('Personal Status and Sex')
plt.ylabel('Proportion')
plt.legend(title='Credit Risk', labels=['Good', 'Bad'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ----------------------------------------
# Chi-Squared Test
# ----------------------------------------

# Create contingency table
contingency_table = pd.crosstab(data['Personal_status_and_sex'], data['Target'])

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display results
print("Chi-Squared Test Results:")
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {dof}")
print("\nExpected frequencies:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

# Interpretation
if p < 0.05:
    print("\nThe p-value is less than 0.05. There is a significant relationship between 'Personal Status and Sex' and credit risk classification (potential bias).")
else:
    print("\nThe p-value is greater than 0.05. There is no significant relationship between 'Personal Status and Sex' and credit risk classification.")
