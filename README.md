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

# Stage 2: CHOOSE TARGET VARIABLE
- Import data:
- Input:
```python
good_status = ['Fully Paid']
bad_status = ['Charged Off', 'Default']

df = df[df['loan_status'].isin(good_status + bad_status)].copy()

df['target'] = np.where(df['loan_status'].isin(bad_status), 1, 0)
```

# Stage 3: CLEAN DATA
## 3.1 Standard data type
- Import data:
- Input:
```python
# Emp length: object → numeric
df['emp_length'] = (
    df['emp_length']
    .astype(str)
    .str.replace('years', '', regex=False)
    .str.replace('year', '', regex=False)
    .str.replace('+', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.strip()
)

df['emp_length'] = df['emp_length'].replace('nan', np.nan).astype(float)
```
## 3.2 Missing values
- Import data:
- Input:
```python
num_cols = [
    'dti', 'delinq_2yrs', 'revol_util',
    'bc_util', 'open_acc', 'inq_last_6mths',
    'emp_length'
]

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
```
## 3.3 Encode categorical variables
- Import data:
- Input:
```python
df = pd.get_dummies(
    df,
    columns=['home_ownership'],
    drop_first=True
)
```
## Outlier detection
- Import data:
- Input:
```python
def cap_outlier(series, lower=0.01, upper=0.99):
    return series.clip(
        lower=series.quantile(lower),
        upper=series.quantile(upper)
    )

outlier_cols = [
    'annual_inc', 'loan_amnt', 'dti',
    'revol_util', 'bc_util',
    'open_acc', 'int_rate', 'emp_length'
]

for col in outlier_cols:
    df[col] = cap_outlier(df[col])
```
## Result after cleaning data
- Import data:
- Input:
```python
df.info()
```

- Output:
  
Index: 267680 entries, 0 to 302139
Data columns (total 15 columns):

| #  | Column                   | Non-Null Count | Dtype   |
|----|--------------------------|----------------|---------|
| 0  | annual_inc               | 267,680        | float64 |
| 1  | loan_amnt                | 267,680        | float64 |
| 2  | term                     | 267,680        | int64   |
| 3  | int_rate                 | 267,680        | float64 |
| 4  | emp_length               | 267,680        | float64 |
| 5  | dti                      | 267,680        | float64 |
| 6  | delinq_2yrs              | 267,680        | float64 |
| 7  | revol_util               | 267,680        | float64 |
| 8  | bc_util                  | 267,680        | float64 |
| 9  | open_acc                 | 267,680        | float64 |
| 10 | inq_last_6mths           | 267,680        | float64 |
| 11 | target                   | 267,680        | int64   |
| 12 | home_ownership_MORTGAGE  | 267,680        | bool    |
| 13 | home_ownership_OWN       | 267,680        | bool    |
| 14 | home_ownership_RENT      | 267,680        | bool    |

Finally, I have resolved the issue with missing values. Now, the dataset is fully prepared for further analysis. 

# Stage 4: CORRELATION & VIF CHECK
## 4.1 Correlation Check
- Import data:
- Input:
```python
# Tính toán tương quan giữa các biến
plt.figure(figsize=(10, 6))
# Thêm numeric_only=True để tránh lỗi ValueError
corr = df.corr(numeric_only=True)
corr
```
- Output:

|                          | annual_inc | loan_amnt | term | int_rate | emp_length | dti | delinq_2yrs | revol_util | bc_util | open_acc | inq_last_6mths | target | home_ownership_MORTGAGE | home_ownership_OWN | home_ownership_RENT |
|--------------------------|------------|-----------|------|----------|------------|-----|-------------|-------------|---------|----------|----------------|--------|--------------------------|--------------------|---------------------|
| annual_inc               | 1.000000 | 0.492016 | 0.071281 | -0.149279 | 0.090892 | -0.246875 | 0.056887 | 0.069321 | 0.005019 | 0.201807 | 0.035461 | -0.064294 | 0.220140 | -0.042266 | -0.197275 |
| loan_amnt                | 0.492016 | 1.000000 | 0.373012 | 0.124990 | 0.082031 | 0.010100 | -0.015437 | 0.117139 | 0.054392 | 0.194861 | -0.027449 | 0.067305 | 0.171152 | -0.007980 | -0.169346 |
| term                     | 0.071281 | 0.373012 | 1.000000 | 0.455480 | 0.046257 | 0.099468 | -0.013828 | 0.061801 | 0.047364 | 0.088230 | 0.022167 | 0.236793 | 0.088598 | -0.013306 | -0.081778 |
| int_rate                 | -0.149279 | 0.124990 | 0.455480 | 1.000000 | -0.017179 | 0.203305 | 0.046083 | 0.191180 | 0.219123 | -0.017004 | 0.247705 | 0.305529 | -0.072015 | 0.006525 | 0.069202 |
| emp_length               | 0.090892 | 0.082031 | 0.046257 | -0.017179 | 1.000000 | 0.036066 | 0.020353 | 0.022600 | 0.017183 | 0.040516 | -0.000256 | -0.008861 | 0.181267 | 0.021218 | -0.198351 |
| dti                      | -0.246875 | 0.010100 | 0.099468 | 0.203305 | 0.036066 | 1.000000 | -0.016525 | 0.171881 | 0.183657 | 0.289159 | 0.009523 | 0.120176 | -0.009398 | 0.027996 | -0.008376 |
| delinq_2yrs              | 0.056887 | -0.015437 | -0.013828 | 0.046083 | 0.020353 | -0.016525 | 1.000000 | -0.016235 | -0.014185 | 0.044072 | 0.040190 | 0.017253 | 0.047868 | -0.006004 | -0.044932 |
| revol_util               | 0.069321 | 0.117139 | 0.061801 | 0.191180 | 0.022600 | 0.171881 | -0.016235 | 1.000000 | 0.839120 | -0.158325 | -0.090875 | 0.053892 | 0.031351 | -0.043599 | -0.004002 |
| bc_util                  | 0.005019 | 0.054392 | 0.047364 | 0.219123 | 0.017183 | 0.183657 | -0.014185 | 0.839120 | 1.000000 | -0.111943 | -0.073384 | 0.064403 | 0.015713 | -0.042494 | 0.011221 |
| open_acc                 | 0.201807 | 0.194861 | 0.088230 | -0.017004 | 0.040516 | 0.289159 | 0.044072 | -0.158325 | -0.111943 | 1.000000 | 0.160876 | 0.042297 | 0.120585 | -0.002471 | -0.121312 |
| inq_last_6mths           | 0.035461 | -0.027449 | 0.022167 | 0.247705 | -0.000256 | 0.009523 | 0.040190 | -0.090875 | -0.073384 | 0.160876 | 1.000000 | 0.090655 | 0.005160 | 0.004426 | -0.008094 |
| target                   | -0.064294 | 0.067305 | 0.236793 | 0.305529 | -0.008861 | 0.120176 | 0.017253 | 0.053892 | 0.064403 | 0.042297 | 0.090655 | 1.000000 | -0.064710 | 0.004790 | 0.062888 |
| home_ownership_MORTGAGE  | 0.220140 | 0.171152 | 0.088598 | -0.072015 | 0.181267 | -0.009398 | 0.047868 | 0.031351 | 0.015713 | 0.120585 | 0.005160 | -0.064710 | 1.000000 | -0.344042 | -0.798671 |
| home_ownership_OWN       | -0.042266 | -0.007980 | -0.013306 | 0.006525 | 0.021218 | 0.027996 | -0.006004 | -0.043599 | -0.042494 | -0.002471 | 0.004426 | 0.004790 | -0.344042 | 1.000000 | -0.290232 |
| home_ownership_RENT      | -0.197275 | -0.169346 | -0.081778 | 0.069202 | -0.198351 | -0.008376 | -0.044932 | -0.004002 | 0.011221 | -0.121312 | -0.008094 | 0.062888 | -0.798671 | -0.290232 | 1.000000 |

# Heat map
- Import data:
- Input:
```python
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation matrix between risk factors")
plt.show()
```

- Output:

<img width="966" height="872" alt="image" src="https://github.com/user-attachments/assets/5834042e-2ca6-4668-a2e5-9f4ff035771d" />





