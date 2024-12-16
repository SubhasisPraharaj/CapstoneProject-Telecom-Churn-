import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file path
file_path = r"C:\Users\archa\Python\telecom_churn_data.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Load the dataset
data = pd.read_csv(file_path)

# Step 1: Identify and handle date or non-numeric columns
# Convert date columns to datetime if necessary
if 'AON' in data.columns:
    data['AON'] = pd.to_numeric(data['AON'], errors='coerce')  # Convert to numeric if it's not already

# Step 2: Filter high-value customers
recharge_cols = ['total_rech_amt_6', 'total_rech_amt_7']
data['avg_rech_amt_good_phase'] = data[recharge_cols].mean(axis=1)
threshold = data['avg_rech_amt_good_phase'].quantile(0.7)
high_value_customers = data[data['avg_rech_amt_good_phase'] >= threshold].copy()  # Explicit copy

# Step 3: Tag churners
churn_conditions = (
    (high_value_customers['total_ic_mou_9'] == 0) &
    (high_value_customers['total_og_mou_9'] == 0) &
    (high_value_customers['vol_2g_mb_9'] == 0) &
    (high_value_customers['vol_3g_mb_9'] == 0)
)
# Use .loc for column assignment
high_value_customers.loc[:, 'churn'] = np.where(churn_conditions, 1, 0)

# Drop churn phase columns
churn_phase_cols = [col for col in high_value_customers.columns if '_9' in col]
high_value_customers.drop(columns=churn_phase_cols, inplace=True)

# Step 4: Prepare data for modeling
columns_to_drop = ['churn', 'mobile_number', 'circle_id'] if 'mobile_number' in high_value_customers.columns else ['churn', 'circle_id']

# Define features (X) and target (y)
X = high_value_customers.drop(columns=columns_to_drop)
y = high_value_customers['churn']

# Keep only numeric columns for modeling
X = X.select_dtypes(include=[np.number]).copy()

# Fill NaNs in numeric columns with 0
X.fillna(0, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build predictive model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

# Step 6: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test, rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_probabilities))

# Step 7: Feature importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Important Features')
plt.show()
