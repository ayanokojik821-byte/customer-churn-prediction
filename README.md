# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset.
The project focuses on clean preprocessing, proper model validation, and reproducible training.

---

## Problem Statement

Customer churn is a critical business problem where retaining existing customers is more cost-effective
than acquiring new ones. The objective of this project is to predict whether a customer will churn
based on demographic, account, and service usage information.

---

## Dataset

- Source: Telco Customer Churn Dataset
- Rows: ~7,000 customers
- Target variable: `Churn` (Yes / No)

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


---

## Approach

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature encoding using one-hot encoding
4. Model training with train-test split
5. Model optimization using cross-validation and hyperparameter tuning
6. Model selection based on ROC-AUC
7. Saving the best-performing model for inference

---

## Model Optimization & Validation

To ensure reliable performance, Stratified 5-Fold Cross-Validation was applied
along with GridSearchCV for hyperparameter tuning.

**Evaluation metric:** ROC-AUC (chosen due to class imbalance)

| Model | CV ROC-AUC | Test ROC-AUC |
|------|-----------|--------------|
| Logistic Regression | 0.81 | 0.79 |
| Random Forest (Tuned) | **0.84** | **0.82** |

The tuned Random Forest model achieved the best generalization performance
and was persisted for downstream inference.

---

## How to Run

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# train model and save best version
python -m src.save_model

