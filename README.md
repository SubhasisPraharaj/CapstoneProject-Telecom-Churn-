Telecom Churn Prediction Project
_______________________________________


This project focuses on predicting customer churn in the telecom industry using machine learning techniques. The goal is to identify high-value customers who are likely to churn and understand the factors contributing to churn. The project involves data preprocessing, exploratory data analysis, feature engineering, and building predictive models using Logistic Regression and Support Vector Machines (SVM).



Table of Contents:-

(i)Project Overview

(ii)Dataset

(iii)Data Preprocessing

(iv)Exploratory Data Analysis (EDA)

(v)Feature Engineering

(vi)Model Building

(vii)Model Evaluation

(viii)Results

(ix)Conclusion

(x)Dependencies

(xi)How to Run



Project Overview:-

The telecom industry faces significant challenges in retaining customers due to high competition. This project aims to predict customer churn by analyzing customer behavior and usage patterns. The project uses a dataset containing customer information, including call details, recharge amounts, and other relevant features.

Dataset:-

The dataset used in this project is telecom_churn_data.csv, which contains customer information over several months. The dataset includes features such as:

total_rech_amt: Total recharge amount
total_mou: Total minutes of usage
arpu: Average revenue per user
churn: Target variable indicating whether the customer churned (1) or not (0)


Data Preprocessing:-

The data preprocessing steps include:
Handling missing values
Dropping columns with more than 30% missing values
Removing date columns and irrelevant features
Identifying high-value customers based on recharge amounts
Tagging churners based on usage in the 9th month
Removing outliers and normalizing data
python
Copy
# Example of data preprocessing
data = data.drop(columns=columns_to_drop)
high_value_customers = data[data['avg_rech_amt_6_7'] >= X]



Exploratory Data Analysis (EDA):-

EDA involves analyzing the distribution of features, identifying patterns, and understanding the relationship between features and churn. Key visualizations include:
Distribution of average revenue per user (ARPU) for churn and non-churn customers
Churn rate by decrease in recharge amount and number of recharges
Scatter plots to visualize the relationship between recharge amount and number of recharges
python
Copy
# Example of EDA
sns.distplot(data_churn['avg_arpu_action'], label='Churn', hist=False, color='red')
sns.distplot(data_non_churn['avg_arpu_action'], label='Not Churn', hist=False, color='green')
Feature Engineering
Feature engineering involves creating new features that capture customer behavior over time. Some of the engineered features include:
Average recharge amount and minutes of usage during the action phase
Difference in recharge amount and usage between the good and action phases
Binary indicators for decreases in recharge amount, usage, and ARPU
python
Copy
# Example of feature engineering
data['avg_rech_amt_action'] = (data['total_rech_amt_7'] + data['total_rech_amt_8']) / 2
data['decrease_rech_amt_action'] = np.where(data['diff_rech_amt'] < 0, 1, 0)




Model Building:-

The project uses two machine learning models:
Logistic Regression: A baseline model to predict churn.
Support Vector Machine (SVM): A more complex model to improve prediction accuracy.
The models are trained on a resampled dataset using SMOTE to handle class imbalance.
python
Copy
# Example of model building
logistic_pca = LogisticRegression(C=best_C)
log_pca_model = logistic_pca.fit(X_train_pca, y_train)



Model Evaluation:-

The models are evaluated using metrics such as accuracy, sensitivity (recall), and specificity. The confusion matrix is used to visualize the performance of the models.
python
Copy
# Example of model evaluation
print("Accuracy:", metrics.accuracy_score(y_train, y_train_pred))
print("Sensitivity:", metrics.recall_score(y_train, y_train_pred))



Results:-

The best-performing model is selected based on the highest recall score. The results show that the SVM model with an RBF kernel performs better than Logistic Regression in predicting churn.
python
Copy
# Example of results
print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))



Conclusion:-

This project successfully identifies high-value customers at risk of churn and provides insights into the factors contributing to churn. The SVM model with an RBF kernel is recommended for predicting churn due to its higher recall score.



Dependencies:-
The project requires the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn imblearn



How to Run:-

Clone the repository.
Install the required dependencies.
Place the dataset telecom_churn_data.csv in the appropriate directory.
Run the Jupyter notebook or Python script.




This README provides an overview of the Telecom Churn Prediction project. For more details, refer to the code and comments in the Jupyter notebook or Python script.