# Bankruptcy Prediction for Polish Companies

Predicts whether a company will go bankrupt about 3 years ahead, using
64 financial ratios. Built with Python, scikit-learn, XGBoost and Streamlit.

## Problem
Banks lose money when a company they lend to goes bankrupt. This project
finds companies at risk from their financial ratios.

## Data
UCI Polish Companies Bankruptcy dataset (3-year file): 10,503 companies,
64 ratios, only 4.7% bankrupt. Download it from the UCI repository and put
`3year.arff` in `data/raw/`.

## What I did
1. Inspected the data: class balance, missing values, duplicates, outliers
2. Split 80/20 (stratified), dropped 1 column that was over 40% empty,
   filled blanks with the median, clipped outliers at the 1st/99th percentile.
   All limits were learned from the training data only (no data leakage)
3. Analysed features: correlation, t-tests, near-duplicate features
4. Compared 7 models: Logistic Regression, Ridge, Lasso, SVM, Decision Tree,
   Random Forest, XGBoost (class weights for the imbalance)
5. Tuned XGBoost with GridSearchCV and 5-fold cross-validation, and chose the
   decision cutoff using training data only
6. Clustering: PCA + K-Means [add k and what you found]
7. Streamlit app that scores a company's risk

## Results (test set)
| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Logistic Regression | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |
| XGBoost (tuned) | [ ] | [ ] | [ ] |

## Key findings
- XGBoost had the best ROC-AUC of the 7 models
- Random Forest ranked companies well but caught few bankruptcies at the
  default 0.50 cutoff, so the cutoff matters as much as the model
- Lower profitability was linked to bankruptcy, but no single feature was enough
- [One finding from your clusters, for example the bankruptcy rate per group]

## Limitations
- The test set has only about 99 bankrupt companies, so small differences
  between models can be noise
- Clusters were weak (silhouette [number]), which is common for financial data
- The app asks for 8 ratios and fills the rest with typical values
- A learning project, not financial advice

## Run it
pip install -r requirements.txt
python3 -m streamlit run app.py

## Project structure
- notebooks/ : step-by-step analysis (01 to 11)
- models/ : saved XGBoost model, settings and demo companies
- app.py : Streamlit app
