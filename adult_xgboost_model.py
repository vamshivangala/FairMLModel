# xgboost_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Fairness libraries
from fairlearn.metrics import (
    MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------
# Load the Processed Data
# ----------------------------------------

# Load the processed data
data = pd.read_csv('adult_data_processed.csv')

# Define the protected attributes
protected_attributes = ['race', 'sex']

# Define features and target
X = data.drop(['income', 'age_group'], axis=1) 
y = data['income']
A = data[protected_attributes]  # A includes 'age_group'

# Function to evaluate and print model metrics
def evaluate_model(y_true, y_pred, model_name):
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
    plt.show()

# Function to calculate and print fairness metrics
def calculate_fairness_metrics(y_true, y_pred, sensitive_features, model_name):
    for attr in sensitive_features.columns:
        print(f"\nFairness Metrics for {model_name} - Protected Attribute: {attr}")

        # Selection rate by group
        sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features[attr])
        print(f"Selection Rate by Group:\n{sr.by_group}")

        # Demographic Parity Difference
        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features[attr])
        print(f"Demographic Parity Difference: {dpd:.4f}")

        # Equalized Odds Difference
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features[attr])
        print(f"Equalized Odds Difference: {eod:.4f}")

# ----------------------------------------
# Baseline Model (Without Blinding)
# ----------------------------------------

print("********** Baseline XGBoost Model **********")

# Include all features (including protected attributes)
X_baseline = X.copy()  # X includes 'race' and 'sex' as features
A_baseline = A.copy()

# Split the data
X_train_bl, X_test_bl, y_train_bl, y_test_bl, A_train_bl, A_test_bl = train_test_split(
    X_baseline, y, A_baseline, test_size=0.2, random_state=42, stratify=y
)

# Train the baseline model
xgb_baseline = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_baseline.fit(X_train_bl, y_train_bl)
y_pred_baseline = xgb_baseline.predict(X_test_bl)

# Evaluate the baseline model
evaluate_model(y_test_bl, y_pred_baseline, "Baseline XGBoost")

# Calculate fairness metrics
calculate_fairness_metrics(y_test_bl, y_pred_baseline, A_test_bl, "Baseline XGBoost")

# ----------------------------------------
# Blinded Model (With Blinding)
# ----------------------------------------

print("\n********** Blinded XGBoost Model **********")

# Redefine protected attributes to only those in X
protected_attributes_in_X = ['race', 'sex']

# Remove protected attributes from features
X_blinded = X.drop(protected_attributes_in_X, axis=1)
A_blinded = A.copy()

# Split the data
X_train_bd, X_test_bd, y_train_bd, y_test_bd, A_train_bd, A_test_bd = train_test_split(
    X_blinded, y, A_blinded, test_size=0.2, random_state=42, stratify=y
)

# Train the blinded model
xgb_blinded = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_blinded.fit(X_train_bd, y_train_bd)
y_pred_blinded = xgb_blinded.predict(X_test_bd)

# Evaluate the blinded model
evaluate_model(y_test_bd, y_pred_blinded, "Blinded XGBoost")

# Calculate fairness metrics
calculate_fairness_metrics(y_test_bd, y_pred_blinded, A_test_bd, "Blinded XGBoost")

# ----------------------------------------
# Fairness-Adjusted Model (Post-processing)
# ----------------------------------------

print("\n********** Fairness-Adjusted XGBoost Model **********")

# Define multiple protected attributes for fairness constraint
sensitive_attributes = ['race', 'sex']

# Initialize the ThresholdOptimizer
constraint = "demographic_parity"  # Or "equalized_odds"
to = ThresholdOptimizer(
    estimator=xgb_baseline,
    constraints=constraint,
    prefit=True,
    predict_method='auto'
)

# Fit the ThresholdOptimizer with multiple sensitive features
to.fit(X_train_bl, y_train_bl, sensitive_features=A_train_bl[sensitive_attributes])

# Make predictions
y_pred_fair = to.predict(X_test_bl, sensitive_features=A_test_bl[sensitive_attributes])

# Evaluate the fairness-adjusted model
evaluate_model(y_test_bl, y_pred_fair, "Fairness-Adjusted XGBoost")

# Calculate fairness metrics
calculate_fairness_metrics(y_test_bl, y_pred_fair, A_test_bl, "Fairness-Adjusted XGBoost")