# Credit-Scoring-Model-for-Predicting-Loan-Default
## Introduction
Credit risk assessment plays a critical role in lending decisions, as inaccurate evaluation of borrower risk can lead to increased default rates and financial losses. With the growing availability of large-scale financial data, data-driven credit scoring models have become an essential tool for financial institutions to evaluate borrower creditworthiness in a consistent and transparent manner.
This project focuses on developing a credit scoring model using historical personal loan data from Lending Club. By applying statistical analysis and machine learning techniques, the project aims to estimate the Probability of Default (PD) of individual borrowers and translate these estimates into a traditional credit scorecard. In addition to predictive modeling, the project emphasizes data visualization to demonstrate how analytical results can be applied in practical credit risk management.
## Dataset
- Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Description: The dataset contains information about .....
- Access: The dataset is loaded directly within the Jupyter Notebook.
## Objectives
The primary purpose of this project is to build a Logistic Regression–based model to predict the Probability of Default for personal loan borrowers based on their financial and loan-related characteristics. The project seeks to examine the relationship between borrower attributes, loan features, and repayment outcomes, and to assess whether these factors can effectively distinguish between low-risk and high-risk customers.
An additional objective is to transform the probabilistic outputs of the model into a credit scorecard that converts statistical results into an interpretable numerical score suitable for business use. Through this process, the project aims to bridge the gap between technical modeling and real-world credit decision-making. The final objective is to visualize the loan portfolio and model performance through interactive dashboards, enabling stakeholders to evaluate risk distribution and model effectiveness in classifying borrowers as “Good” or “Bad”.
# Stage 1: DATA IMPORT
- Import data:
- Input:
```python
#Import the libraries

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Import the dataset into dataframe (df)
df = pd.read_csv("/content/accepted_2007_to_2018Q4.csv")

# Display the first few rows of the dataset
df.head()
```

- Output:

| idx | id       | member_id | loan_amnt | funded_amnt | funded_amnt_inv | term       | int_rate | installment | grade | sub_grade | disbursement_method | debt_settlement_flag |
|-----|----------|-----------|-----------|-------------|------------------|------------|----------|-------------|-------|-----------|----------------------|----------------------|
| 0   | 68407277 | NaN       | 3600.0    | 3600.0      | 3600.0           | 36 months  | 13.99    | 123.03      | C     | C4        | Cash                 | N                    |
| 1   | 68355089 | NaN       | 24700.0   | 24700.0     | 24700.0          | 36 months  | 11.99    | 820.28      | C     | C1        | Cash                 | N                    |
| 2   | 68341763 | NaN       | 20000.0   | 20000.0     | 20000.0          | 60 months  | 10.78    | 432.66      | B     | B4        | Cash                 | N                    |
| 3   | 66310712 | NaN       | 35000.0   | 35000.0     | 35000.0          | 60 months  | 14.85    | 829.90      | C     | C5        | Cash                 | N                    |
| 4   | 68476807 | NaN       | 10400.0   | 10400.0     | 10400.0          | 60 months  | 22.45    | 289.91      | F     | F1        | Cash                 | N                    |




# Stage 1: DATA CLEANING




