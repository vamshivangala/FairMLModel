# logistic_regression_fairness_combined.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import Fairlearn metrics and algorithms
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------
# Load the Processed Data
# ----------------------------------------

# Load the processed data
data = pd.read_csv('german_credit_processed.csv')

# Define the protected attribute
protected_attribute = 'Personal_status_and_sex'  # You can change this as needed

# Function to evaluate and print model metrics
def evaluate_model(y_true, y_pred, model_name, A_test):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['Good Credit', 'Bad Credit'])
    plt.yticks([0.5, 1.5], ['Good Credit', 'Bad Credit'])
    plt.tight_layout()
    plt.show()

    # Calculate fairness metrics
    calculate_fairness_metrics(y_true, y_pred, A_test, model_name)

# Function to calculate and print fairness metrics
def calculate_fairness_metrics(y_true, y_pred, sensitive_features, model_name):
    # Calculate selection rate
    sr = selection_rate(y_true, y_pred)

    # Create a MetricFrame for selection rate
    sr_frame = MetricFrame(metrics=selection_rate,
                           y_true=y_true,
                           y_pred=y_pred,
                           sensitive_features=sensitive_features)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    print(f"\nFairness Metrics for {model_name}:")
    print(f"Overall Selection Rate: {sr:.4f}")
    print(f"Selection Rate by Group:")
    print(sr_frame.by_group)
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")

# ----------------------------------------
# Baseline Model (Without Blinding)
# ----------------------------------------

print("\n********** Baseline Model (Without Blinding) **********")

# Separate features, target, and protected attribute
X = data.drop('Target', axis=1)
y = data['Target']
A = data[protected_attribute]

# Split the data (including protected attributes)
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.2, stratify=y, random_state=42
)

# Initialize and train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Logistic Regression model
evaluate_model(y_test, y_pred_lr, "Logistic Regression (Baseline)", A_test)

# ----------------------------------------
# Blinded Model (With Blinding)
# ----------------------------------------

print("\n********** Blinded Model (With Blinding) **********")

# Remove protected attributes from features
protected_attributes = [protected_attribute]  # You can add more attributes to this list if needed
X_blinded = data.drop(['Target'] + protected_attributes, axis=1)
y_blinded = data['Target']
A_blinded = data[protected_attribute]  # Keep for fairness metrics

# Split the data
X_train_blind, X_test_blind, y_train_blind, y_test_blind, A_train_blind, A_test_blind = train_test_split(
    X_blinded, y_blinded, A_blinded, test_size=0.2, stratify=y_blinded, random_state=42
)

# Initialize and train Logistic Regression model (blinded)
lr_model_blind = LogisticRegression(max_iter=1000)
lr_model_blind.fit(X_train_blind, y_train_blind)
y_pred_lr_blind = lr_model_blind.predict(X_test_blind)

# Evaluate Logistic Regression model (blinded)
evaluate_model(y_test_blind, y_pred_lr_blind, "Logistic Regression (Blinded)", A_test_blind)

# ----------------------------------------
# In-processing Model (With Fairness Adjustment)
# ----------------------------------------

print("\n********** In-processing Model (With Fairness Adjustment) **********")

# Initialize the base estimator
estimator = LogisticRegression(max_iter=1000, solver='liblinear')

# Define the fairness constraint
constraint = DemographicParity()  # Or use EqualizedOdds()

# Initialize the Exponentiated Gradient Reduction algorithm
eg = ExponentiatedGradient(estimator=estimator, constraints=constraint)

# Fit the model with fairness constraint
eg.fit(X_train, y_train, sensitive_features=A_train)

# Make predictions
y_pred_eg = eg.predict(X_test)

# Evaluate the model
evaluate_model(y_test, y_pred_eg, "Logistic Regression (In-processing Fairness Adjustment)", A_test)

