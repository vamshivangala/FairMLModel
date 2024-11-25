import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
data = pd.read_csv('adult_data_processed.csv')

# Load encodings for categorical variables
import pickle
with open('adult_categorical_encodings.pkl', 'rb') as f:
    encodings = pickle.load(f)

# Reverse map categorical variables for visualization
for col, mapping in encodings.items():
    reverse_mapping = {v: k for k, v in mapping.items()}
    data[col] = data[col].map(reverse_mapping)

# Reverse map the target variable
data['income'] = data['income'].map({0: '<=50K', 1: '>50K'})


# Define colors
colors = ['red', 'blue']

# ----------------------------------------
# Income Distribution by Race
# ----------------------------------------

# Compute proportions for race and income
race_income = data.groupby(['race', 'income']).size().unstack(fill_value=0)
race_income_ratio = race_income.div(race_income.sum(axis=1), axis=0)  # Normalize rows

# Plot stacked bar chart
race_income_ratio.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), edgecolor='black')
plt.title('Income Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Proportion')
plt.legend(title='Income', labels=['<=50K', '>50K'])
plt.tight_layout()
plt.show()

# ----------------------------------------
# Income Distribution by Gender
# ----------------------------------------

# Compute proportions for gender and income
gender_income = data.groupby(['sex', 'income']).size().unstack(fill_value=0)
gender_income_ratio = gender_income.div(gender_income.sum(axis=1), axis=0)  # Normalize rows

# Plot stacked bar chart
gender_income_ratio.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6), edgecolor='black')
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.legend(title='Income', labels=['<=50K', '>50K'])
plt.tight_layout()
plt.show()

# ----------------------------------------
# Education-Num vs Income
# ----------------------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='education-num', data=data, palette='Set2', showfliers=False)
plt.title('Education Level Distribution by Income')
plt.xlabel('Income')
plt.ylabel('Education Level (Education-Num)')
plt.tight_layout()
plt.show()

# ----------------------------------------
# Education-Num vs Race
# ----------------------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(x='race', y='education-num', data=data, palette='Set3', showfliers=False)
plt.title('Education Level Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Education Level (Education-Num)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------
# Hours-Per-Week vs Income
# ----------------------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='hours-per-week', data=data, palette='coolwarm', showfliers=False)
plt.title('Work Hours Distribution by Income')
plt.xlabel('Income')
plt.ylabel('Hours Per Week')
plt.tight_layout()
plt.show()

# ----------------------------------------
# Hours-Per-Week vs Gender
# ----------------------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='hours-per-week', data=data, palette='viridis', showfliers=False)
plt.title('Work Hours Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Hours Per Week')
plt.tight_layout()
plt.show()
