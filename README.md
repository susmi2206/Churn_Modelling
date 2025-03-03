# Churn Modelling

## Overview  
This project aims to predict **customer churn** using various machine learning models. The dataset includes customer details such as credit score, geography, age, tenure, balance, and account activity, which help determine whether a customer is likely to leave the service.

## Dataset  
The dataset used is **Churn_Modelling.csv**, containing the following key columns:

- `RowNumber`, `CustomerId`, `Surname` (Ignored for training)
- `CreditScore` (Numeric)
- `Geography` (Categorical: Encoded)
- `Gender` (Categorical: Encoded)
- `Age`, `Tenure`, `Balance` (Numeric features)
- `NumOfProducts`, `HasCrCard`, `IsActiveMember` (Customer-related features)
- `EstimatedSalary` (Numeric feature)
- `Exited` (Target variable: `1` if customer churned, `0` otherwise)

## Preprocessing Steps  
- **Handle Missing Values**: Remove any missing data.  
- **Encoding**: Convert categorical features (`Geography`, `Gender`) using `LabelEncoder`.  
- **Feature Selection**: Drop unnecessary columns (`RowNumber`, `CustomerId`, `Surname`).  
- **Data Splitting**: Train-test split with `80%-20%` ratio.  
- **Feature Scaling**: Apply `StandardScaler` for numerical features.  
- **Data Balancing**: Use **SMOTE** to handle class imbalance.  

## Machine Learning Models  
The following models were trained and evaluated using **classification reports** and **ROC AUC scores**:

1. **Logistic Regression**
2. **Random Forest Classifier** (with `GridSearchCV` for hyperparameter tuning)
3. **Gradient Boosting Classifier**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)** (Tested for different `n_neighbors`)
6. **Naive Bayes (GaussianNB)**
7. **Artificial Neural Network (MLP Classifier)**
8. **XGBoost Classifier**
9. **AdaBoost Classifier**

## Installation  
To install the required dependencies, run:

```bash
pip install pandas scikit-learn imbalanced-learn xgboost
