# Creditworthiness Prediction for Credit Card Issuance

Binary classification model to predict customer creditworthiness, built for a fictional bank (Pro National Bank) to support credit card application decisions.

## Overview

Given anonymized customer data, the model predicts whether a customer has a high credit rating (TARGET=1) or not (TARGET=0). The task presents a significant class imbalance: only 8.78% of observations belong to the positive class.

The project evaluates **8 base classifiers** across **6 feature selection strategies**, resulting in 291 model configurations compared systematically.

## Dataset

~338,000 customer records with 19 features including demographics, employment, income, and contact information.

```
https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv
```

**Key preprocessing decisions:**
- `DAYS_BIRTH` converted to positive values (`DAYS_BIRTH_POSITIVE`)
- `FLAG_MOBIL` dropped - 100% of values identical, zero predictive value
- `DAYS_EMPLOYED` anomalies handled - values exceeding human lifespan (~1,000 years) identified as Pensioners and treated separately
- `OCCUPATION_TYPE` NaN values (13.44% of data) replaced with `"unknown"` - neither deletion nor imputation was justified after systematic analysis across all other features

## Pipeline

**Preprocessing (ColumnTransformer):**
- Numerical features → StandardScaler
- Nominal categorical → OneHotEncoder
- Ordinal categorical (`NAME_EDUCATION_TYPE`) → OrdinalEncoder with explicit category ordering

**Feature selection strategies evaluated:**
1. Base (all features)
2. VarianceThreshold
3. Correlation Filtering
4. VIF (Variance Inflation Factor)
5. SelectKBest (k = 5, 10, 15, 20, 25, 30, 35, 40)
6. RFE - Recursive Feature Elimination (k = 5–40)
7. Embedded Methods - hyperparameter tuning with RandomizedSearchCV (regularization, tree depth, class weighting)

## Models

291 configurations evaluated (8 classifiers × 6 feature selection strategies + hyperparameter tuning). Final ranking filtered by Accuracy > 0.90 and sorted by Recall to minimize false negatives - failing to identify a creditworthy customer is the costliest error in this context.

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | FN |
|---|---|---|---|---|---|---|
| Random Forest emb | 0.96 | 0.823 | 0.699 | **1.000** | 0.979 | **0** |
| Gradient Boosting rfe k:35 | 0.96 | 0.823 | 0.699 | 0.999 | 0.979 | 5 |
| Gradient Boosting CorrelationFiltering | 0.96 | 0.823 | 0.699 | 0.999 | 0.979 | 5 |
| Gradient Boosting rfe k:30 | 0.96 | 0.823 | 0.699 | 0.999 | 0.979 | 5 |
| Gradient Boosting rfe k:20 | 0.96 | 0.823 | 0.699 | 0.999 | 0.979 | 5 |

*Top 5 of 291 configurations. Full results in `sorted_results_df.csv`.*

**Best model: Random Forest (Embedded)** - the only configuration achieving perfect Recall (1.000, **0 false negatives**) with F1 = 0.823. Total training time: ~226 minutes.


## Key Findings

- Tree-based ensemble methods (GBM, Random Forest, XGBoost) consistently outperform linear models on this dataset
- Feature selection strategies did not significantly improve performance over the base configuration - the full feature set was already informative
- Logistic Regression achieves high Recall only when Accuracy is sacrificed (0.828), making it unsuitable for production despite its interpretability

## Tech Stack

Python 3.9 · pandas 2.2.2 · NumPy 2.0.0 · scikit-learn 1.5.1 · XGBoost 2.1.0 · SciPy 1.13.1 · statsmodels 0.14.2 · Matplotlib 3.9.1 · Seaborn 0.13.2 · tqdm 4.66.4

## How to Run

The notebook runs in a **local Anaconda environment**.

```bash
conda env create -f environment.yml
conda activate creditworthiness_project
jupyter notebook
```

Download the dataset manually and place it in the project folder:
```
https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv
```

Open `Prediction_of_creditworthiness_for_issuing_a_credit_card.ipynb` and run cells sequentially.

## Project Structure

```
creditworthiness/
├── Prediction_of_creditworthiness_for_issuing_a_credit_card.ipynb   # Main notebook
├── environment.yml                                                    # Conda environment
├── results_df.csv                                                     # All 401 model results
└── sorted_results_df.csv                                              # Final ranked results
```
