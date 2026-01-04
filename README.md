# Customer Churn Prediction (Telecom)

## Overview
This project implements an end-to-end machine learning pipeline to predict customer churn for a telecom business. The goal is to identify customers who are likely to leave the service so that retention efforts can be targeted effectively, rather than applied uniformly across the customer base.

The project emphasizes:
- Handling imbalanced classification problems  
- Proper model validation and comparison  
- Reproducible and modular training code suitable for extension  

---

## Problem Statement
Customer churn has a direct impact on revenue and customer lifetime value in subscription-based businesses. Instead of reacting after customers leave, businesses benefit from predicting churn in advance and intervening with targeted retention strategies.

The objective of this project is to build a binary classification model that predicts whether a customer will churn based on demographic, account, and service usage information. Model performance is evaluated primarily using **ROC-AUC** to account for class imbalance.

---

## Dataset
- **Source:** IBM Telco Customer Churn Dataset (public dataset)
- **Observations:** 7,043 customers
- **Target Variable:** `Churn` (Yes / No)
- **Feature Types:**
  - Demographic information  
  - Account and contract details  
  - Service usage indicators  

Basic data cleaning was required to handle missing values and inconsistent data types.

---

## Project Structure
customer-churn-prediction/
├── data/
│ ├── raw/
│ │ └── telco_churn.csv
│ └── processed/
├── models/
│ └── churn_model.pkl
├── notebooks/
│ └── eda.ipynb
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ └── save_model.py
├── README.md
├── requirements.txt
└── .gitignore


The project is organized to separate exploratory analysis, preprocessing, training, and model persistence, enabling reproducibility and easier maintenance.

---

## Approach

### Data Cleaning & Preprocessing
- Handled missing values and invalid entries  
- Converted categorical variables into numerical form using one-hot encoding  

### Exploratory Data Analysis (EDA)
- Analyzed churn distribution and key feature relationships  
- Identified patterns related to contract type, tenure, and service usage  

### Model Training
- Applied a stratified train-test split to preserve class distribution  
- Trained baseline and ensemble models for comparison  

### Model Optimization & Validation
- Used Stratified 5-Fold Cross-Validation  
- Performed hyperparameter tuning with GridSearchCV  
- Selected models based on ROC-AUC due to class imbalance  

### Model Persistence
- Saved the best-performing model for reuse in downstream inference workflows  

---

## Model Evaluation
**Primary Metric:** ROC-AUC (chosen due to class imbalance)

| Model | CV ROC-AUC | Test ROC-AUC |
|------|-----------|--------------|
| Logistic Regression | 0.81 | 0.79 |
| Random Forest (Tuned) | 0.84 | 0.82 |

The tuned Random Forest model demonstrated stronger generalization performance across both cross-validation and held-out test data and was selected as the final model.

---

## Limitations
- The dataset represents a single historical snapshot and does not capture temporal churn behavior.
- The model does not incorporate cost-sensitive learning, where false negatives may be more expensive than false positives.
- Results may not generalize to other telecom markets without retraining and validation.

---

## Future Improvements
- Introduce time-based validation to better simulate real-world deployment.
- Explore advanced imbalance handling techniques such as class-weighted loss functions.
- Deploy the model via a REST API for real-time inference.
- Perform feature importance analysis to improve business interpretability.

---

## How to Run
```bash
pip install -r requirements.txt
python src/train.py

