# xgboost_fairness_combined.py

# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import Fairlearn
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load the processed data
data = pd.read_csv('german_credit_processed.csv')

# Define the protected attribute
protected_attribute = 'Personal_status_and_sex'  # You can choose other protected attributes

# Function to evaluate and print model metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{model_name} Model Performance:")
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
    print(f"Selection Rate by Group:\n{sr_frame.by_group}")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")

# Function to train and evaluate a model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, A_train, A_test, model_name):
    # Initialize and train the XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred, model_name)

    # Calculate fairness metrics
    calculate_fairness_metrics(y_test, y_pred, A_test, model_name)

    return xgb_model, y_pred

# Function to apply fairness adjustments using ThresholdOptimizer
def apply_fairness_adjustment(estimator, X_train, y_train, X_test, y_test, A_train, A_test, constraint, model_name):
    # Initialize the ThresholdOptimizer
    to = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        prefit=True,
        predict_method='auto',
        grid_size=1000
    )

    # Fit the ThresholdOptimizer
    to.fit(X_train, y_train, sensitive_features=A_train)

    # Make predictions with fairness adjustments
    y_pred_fair = to.predict(X_test, sensitive_features=A_test)

    # Evaluate the fairness-adjusted model
    evaluate_model(y_test, y_pred_fair, model_name)

    # Calculate fairness metrics
    calculate_fairness_metrics(y_test, y_pred_fair, A_test, model_name)

    return y_pred_fair

# ----------------------------------------
# Baseline Model (Without Blinding)
# ----------------------------------------

print("=== Baseline Model (Without Blinding) ===")

# Separate features and target (include protected attribute)
X_baseline = data.drop(['Target'], axis=1)
y = data['Target']
A = data[protected_attribute]

# Split the data
X_train_bl, X_test_bl, y_train_bl, y_test_bl, A_train_bl, A_test_bl = train_test_split(
    X_baseline, y, A, test_size=0.2, stratify=y, random_state=42
)

# Train and evaluate the Baseline model
baseline_model, y_pred_baseline = train_and_evaluate_model(
    X_train_bl, X_test_bl, y_train_bl, y_test_bl, A_train_bl, A_test_bl, "Baseline XGBoost"
)

# ----------------------------------------
# Blinded Model (With Blinding)
# ----------------------------------------

print("\n=== Blinded Model (With Blinding) ===")

# Remove protected attribute from features
X_blinded = data.drop(['Target', protected_attribute], axis=1)

# Split the data
X_train_bd, X_test_bd, y_train_bd, y_test_bd, A_train_bd, A_test_bd = train_test_split(
    X_blinded, y, A, test_size=0.2, stratify=y, random_state=42
)

# Train and evaluate the Blinded model
blinded_model, y_pred_blinded = train_and_evaluate_model(
    X_train_bd, X_test_bd, y_train_bd, y_test_bd, A_train_bd, A_test_bd, "Blinded XGBoost"
)

# ----------------------------------------
# Fairness-Adjusted Model (Post-Processing)
# ----------------------------------------

print("\n=== Fairness-Adjusted Model (Post-Processing) ===")

# Apply fairness adjustment to the baseline model
constraint = "demographic_parity"  # or "equalized_odds"
y_pred_fair = apply_fairness_adjustment(
    baseline_model, X_train_bl, y_train_bl, X_test_bl, y_test_bl, A_train_bl, A_test_bl, constraint, "Fairness-Adjusted XGBoost"
)

# ----------------------------------------
# Comparison of Results
# ----------------------------------------

# Collect metrics for comparison
results = {
    'Model': ['Baseline', 'Blinded', 'Fairness-Adjusted'],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'DPD': [],  # Demographic Parity Difference
    'EOD': []   # Equalized Odds Difference
}

# Function to collect metrics
def collect_metrics(y_true, y_pred, sensitive_features):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return accuracy, precision, recall, f1, dpd, eod

# Collect metrics for Baseline model
metrics_baseline = collect_metrics(y_test_bl, y_pred_baseline, A_test_bl)
# Collect metrics for Blinded model
metrics_blinded = collect_metrics(y_test_bd, y_pred_blinded, A_test_bd)
# Collect metrics for Fairness-Adjusted model
metrics_fair = collect_metrics(y_test_bl, y_pred_fair, A_test_bl)

# Append metrics to results
for metrics in [metrics_baseline, metrics_blinded, metrics_fair]:
    results['Accuracy'].append(metrics[0])
    results['Precision'].append(metrics[1])
    results['Recall'].append(metrics[2])
    results['F1 Score'].append(metrics[3])
    results['DPD'].append(metrics[4])
    results['EOD'].append(metrics[5])

# Create a DataFrame for comparison
results_df = pd.DataFrame(results)

print("\n=== Comparison of Models ===")
print(results_df)
