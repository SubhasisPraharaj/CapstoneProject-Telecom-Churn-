Telecom Churn  Project

Overview

This project analyzes a telecom dataset to identify high-value customers and predict customer churn using machine learning models. The script performs data preprocessing, builds predictive models, evaluates their performance, and provides insights on the key factors influencing churn.

Features

High-Value Customer Identification: Filters customers based on recharge amount during a specific period.

Churn Tagging: Identifies churners based on usage behavior.

Machine Learning Models: Implements a Random Forest Classifier for churn prediction.

Feature Importance Analysis: Highlights the most significant predictors of churn.


Business Objective

The goal of this analysis is to identify high-value customers and understand churn behavior in the telecom sector. By leveraging data-driven insights, the aim is to develop predictive models that help:

(i)Improve customer retention.
(ii)Increase revenue by targeting high-value customers.
(iii)Optimize marketing and operational strategies.




Visualization: Displays feature importance using bar plots.

Prerequisites

Libraries

Ensure the following Python libraries are installed:

pandas

numpy

scikit-learn

matplotlib

seaborn

Install these using pip if not already installed:

pip install pandas numpy scikit-learn matplotlib seaborn

Dataset

The script requires a dataset named telecom_churn_data.csv to be present in the working directory. If the dataset is in a different location, update the file_path variable in the script to the correct path.

Usage

Clone or download the project files.

Place the dataset (telecom_churn_data.csv) in the same directory as the script.

Run the script using the command:

python main.py

Script Breakdown

1. Data Loading

The script loads the dataset and verifies its presence:

file_path = 'telecom_churn_data.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
data = pd.read_csv(file_path)

2. High-Value Customer Identification

Calculates the average recharge amount during two months and selects customers above the 70th percentile:

recharge_cols = ['total_rech_amt_6', 'total_rech_amt_7']
data['avg_rech_amt_good_phase'] = data[recharge_cols].mean(axis=1)
threshold = data['avg_rech_amt_good_phase'].quantile(0.7)
high_value_customers = data[data['avg_rech_amt_good_phase'] >= threshold]

3. Churn Tagging

Tags customers as churners based on their inactivity during a specific period:

churn_conditions = (
    (high_value_customers['total_ic_mou_9'] == 0) &
    (high_value_customers['total_og_mou_9'] == 0) &
    (high_value_customers['vol_2g_mb_9'] == 0) &
    (high_value_customers['vol_3g_mb_9'] == 0)
)
high_value_customers['churn'] = np.where(churn_conditions, 1, 0)

4. Data Preparation for Modeling

Prepares the dataset by handling missing values, splitting into training and testing sets, and scaling features:

X = high_value_customers.drop(columns=['churn', 'MOBILE_NUMBER'])
y = high_value_customers['churn']
X = X.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

5. Model Training and Evaluation

Builds a Random Forest Classifier, makes predictions, and evaluates the model:

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

Evaluation:

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test, rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_probabilities))

6. Feature Importance Analysis

Analyzes the importance of features in predicting churn:

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Important Features')
plt.show()

Output

Confusion Matrix: Displays the classification results.

Classification Report: Provides precision, recall, F1-score, and support metrics.

ROC-AUC Score: Measures the model's ability to distinguish between classes.

Feature Importance Plot: Highlights the most influential features in the prediction.

Notes

Ensure the dataset contains the required columns (e.g., total_rech_amt_6, total_rech_amt_7, etc.).

Replace placeholders with actual paths or column names if required.

Update hyperparameters of the Random Forest model for optimization.