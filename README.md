# Credit Risk Assessment with Explainable Machine Learning

An end-to-end credit risk modelling project that predicts loan default risk, compares interpretable and gradient boosting models, calibrates predicted probabilities, and explains individual credit decisions using SHAP.

**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Joblib

## Problem Statement

Lenders need to identify applicants who are likely to default while avoiding unnecessary rejection of reliable customers.

This creates two important risks:

- Approving a high-risk applicant can lead to financial loss
- Rejecting a reliable applicant can result in lost business opportunities

The goal of this project is to develop a credit risk assessment pipeline that can:

- Predict whether a borrower is likely to default
- Estimate the probability of default
- Identify high-risk applicants
- Handle imbalanced credit data
- Compare an interpretable baseline with a stronger nonlinear model
- Produce more reliable probability estimates through calibration
- Explain the factors behind individual predictions
- Analyse false positives and false negatives from a business perspective

The project uses Logistic Regression as an interpretable baseline and XGBoost as the main predictive model.

## Business Objective

A useful credit risk model should not only classify applicants correctly.

It should also help answer:

- How likely is this applicant to default?
- Which characteristics are contributing to the predicted risk?
- How confident should we be in the predicted probability?
- What types of applicants are being incorrectly classified?
- What is the potential impact of false approvals and false rejections?

This project approaches credit risk as both a predictive modelling and model interpretability problem.

## Key Results

The tuned XGBoost model achieved the strongest performance on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.82 | 0.56 | 0.79 | 0.65 |
| XGBoost | **0.92** | **0.81** | **0.81** | **0.81** |

Five-fold stratified cross-validation also showed a clear improvement over the Logistic Regression baseline.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.871 | 0.812 | 0.545 | 0.778 | 0.641 |
| XGBoost | **0.939** | **0.909** | **0.791** | **0.786** | **0.788** |

The tuned XGBoost model achieved an average precision score of approximately **0.90** during hyperparameter search.

## Dataset

The dataset contains **32,581 loan applications** with information describing the applicant, loan characteristics, and credit history.

The target variable is:

`loan_status`

where:

- `0` represents non-default
- `1` represents default

The original class distribution contains:

- 25,473 non-default observations
- 7,108 default observations

This creates a class imbalance that must be considered during model training and evaluation.

### Features

The model uses information such as:

- Applicant age
- Annual income
- Employment length
- Home ownership
- Loan purpose
- Loan grade
- Loan amount
- Interest rate
- Loan-to-income ratio
- Previous default history
- Credit history length

These variables provide information about both the applicant's financial situation and the characteristics of the requested loan.

## Project Workflow

```text
Credit Risk Data
      |
      v
Data Validation
      |
      v
Exploratory Data Analysis
      |
      v
Data Cleaning
      |
      v
Stratified Train-Test Split
      |
      v
Preprocessing Pipeline
      |
      v
Class Imbalance Handling
      |
      v
Logistic Regression Baseline
      |
      v
XGBoost Model
      |
      v
Stratified Cross-Validation
      |
      v
Hyperparameter Tuning
      |
      v
Probability Calibration
      |
      v
SHAP Explainability
      |
      v
Error Analysis
      |
      v
Model Persistence
```

## Data Validation and Cleaning

Before training the models, the dataset is checked for data quality problems that could affect model performance.

The validation process includes:

- Missing value inspection
- Duplicate detection
- Unrealistic applicant ages
- Employment length checks
- Employment history relative to applicant age
- Extreme numerical observations
- Loan amount validation
- Interest rate inspection
- Class distribution analysis

Cleaning these observations helps prevent unrealistic records from influencing the models.

## Exploratory Data Analysis

Exploratory analysis is used to understand how applicant and loan characteristics differ between default and non-default cases.

The analysis focuses on variables such as:

- Income
- Loan amount
- Interest rate
- Loan-to-income ratio
- Loan grade
- Loan purpose
- Home ownership
- Previous default history

The objective is not only to visualize the data but also to understand which characteristics may contain useful information for credit risk prediction.

## Train-Test Strategy

The cleaned dataset is separated into training and test data using a stratified split.

Stratification keeps the proportion of default and non-default observations similar across both datasets.

This is particularly important because the target variable is imbalanced.

The test set remains separate from model training and hyperparameter selection so that it can provide an independent evaluation of final model performance.

## Preprocessing

The preprocessing steps are implemented using Scikit-learn `Pipeline` and `ColumnTransformer`.

This keeps transformation logic connected to the model and ensures that the same preprocessing steps are applied consistently during both training and prediction.

### Numerical Features

Numerical variables are handled using:

- Median imputation for missing values
- Standard scaling where required by the model

### Categorical Features

Categorical variables are handled using:

- Missing value imputation
- One-hot encoding

Different preprocessing strategies are used for Logistic Regression and XGBoost because the models have different requirements.

## Logistic Regression Baseline

Logistic Regression is used as the baseline model.

It provides an interpretable reference that makes it possible to determine whether a more complex machine learning model produces a meaningful performance improvement.

For Logistic Regression:

- Numerical variables are median-imputed
- Numerical features are standardized
- Categorical variables are one-hot encoded
- Balanced class weights are used

### Baseline Performance

The Logistic Regression model achieved approximately:

- Accuracy: **82%**
- Precision: **56%**
- Recall: **79%**
- F1 Score: **65%**

The relatively high recall means the model identifies many default cases.

However, its lower precision indicates that a larger number of reliable applicants are also classified as risky.

This establishes a useful baseline for evaluating the XGBoost model.

## Handling Class Imbalance

Loan defaults represent a smaller proportion of the dataset.

A model trained without considering this imbalance could achieve reasonable accuracy simply by favouring the majority non-default class.

Different strategies are therefore used for the two models.

### Logistic Regression

Balanced class weights are used so that errors on the minority default class receive more importance during training.

### XGBoost

The `scale_pos_weight` parameter is calculated using the ratio between non-default and default observations in the training data.

The resulting value is approximately:

```text
3.63
```

This increases the importance of default observations during model training.

## XGBoost

XGBoost is used as the main nonlinear model.

Unlike Logistic Regression, tree-based boosting can capture:

- Nonlinear relationships
- Interactions between variables
- Complex decision boundaries

These characteristics are useful in credit risk because borrower behaviour is unlikely to depend on completely linear relationships.

## Hyperparameter Tuning

The XGBoost model is optimized using `RandomizedSearchCV`.

The search evaluates **150 hyperparameter combinations** using **5-fold stratified cross-validation**.

The optimization metric is:

**Average Precision**

Average Precision is particularly useful for this problem because the target classes are imbalanced and the model needs to identify the smaller default class effectively.

The search explores parameters including:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`

This allows the model to balance predictive performance and model complexity.

## Cross-Validation

A single train-test split can produce results that depend on one particular division of the data.

To obtain a more reliable estimate of model performance, both models are evaluated using **5-fold Stratified Cross-Validation**.

The following metrics are measured:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score

### Cross-Validation Results

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.871 | **0.939** |
| Accuracy | 0.812 | **0.909** |
| Precision | 0.545 | **0.791** |
| Recall | 0.778 | **0.786** |
| F1 Score | 0.641 | **0.788** |

XGBoost provides the strongest overall balance between default detection and prediction precision.

## Final XGBoost Performance

On the held-out test set, the tuned XGBoost model achieved:

- **92% Accuracy**
- **81% Precision**
- **81% Recall**
- **81% F1 Score**

The results show a substantial improvement over the Logistic Regression baseline, particularly in precision and F1 score.

This means the model is able to identify default cases while generating fewer false alarms.

## Why Accuracy Alone Is Not Enough

Accuracy can be misleading in credit risk because most borrowers belong to the non-default class.

For this reason, model evaluation also includes:

### Precision

Among applicants predicted to default, how many actually defaulted?

Higher precision reduces the number of reliable applicants incorrectly classified as risky.

### Recall

Among applicants who actually defaulted, how many did the model identify?

Higher recall reduces the number of risky applicants that the model fails to detect.

### F1 Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures the model's ability to rank risky applicants above lower-risk applicants across different classification thresholds.

### Average Precision

Evaluates performance across the precision-recall curve and is useful when working with imbalanced classes.

## Probability Calibration

Credit risk decisions often depend on probabilities rather than only binary predictions.

For example:

```text
Applicant A: 12% predicted probability of default
Applicant B: 67% predicted probability of default
```

These probabilities can support risk ranking and further decision processes.

However, a model's raw probability estimates are not automatically well calibrated.

A predicted probability of 30% should ideally correspond to a group of applicants where approximately 30% actually default.

The XGBoost probabilities are therefore calibrated using:

`CalibratedClassifierCV`

with sigmoid calibration and five-fold cross-validation.

Calibration curves are used to compare the original and calibrated probabilities.

This step focuses on making the model's risk estimates more meaningful rather than relying only on hard classifications.

## Explainable AI with SHAP

High predictive performance is not enough for a credit risk model.

It is also important to understand why the model considers an applicant risky.

SHAP is used to interpret the XGBoost model at both the overall model level and individual applicant level.

## Global SHAP Explainability

SHAP summary plots are used to understand which variables have the greatest overall influence on model predictions.

This helps answer:

- Which variables influence default predictions most strongly?
- Which feature values are associated with increasing predicted risk?
- Which characteristics tend to reduce predicted risk?
- How does the model use applicant and loan information?

Instead of treating XGBoost as a black box, SHAP provides a way to inspect how the model reaches its predictions.

## Applicant-Level Explainability

Global explanations describe the model as a whole.

Credit decisions also require understanding individual predictions.

SHAP waterfall plots are therefore used to explain individual applicants.

For each prediction, SHAP shows:

```text
Model baseline risk
        +
Applicant-specific feature contributions
        =
Final model prediction
```

Some characteristics push the predicted risk higher while others reduce it.

This provides a clear explanation of why the model assigns a particular risk level to an applicant.

## Error Analysis

Model evaluation continues beyond overall performance metrics.

The tuned XGBoost model produced:

- **258 false positives**
- **256 false negatives**

These errors have different business meanings.

### False Positive

The model predicts that an applicant will default, but the applicant actually does not default.

Potential consequence:

A reliable applicant may be incorrectly considered high risk.

This could result in:

- Lost customers
- Reduced lending opportunities
- Lower potential revenue

### False Negative

The model predicts that an applicant will not default, but the applicant actually defaults.

Potential consequence:

The lender may approve a risky loan.

This can result in:

- Credit losses
- Increased portfolio risk
- Additional collection costs

Analysing these two types of errors helps connect model evaluation with the real business consequences of credit decisions.

## Model Comparison

| Area | Logistic Regression | XGBoost |
|---|---|---|
| Interpretability | High | Lower without explanation tools |
| Nonlinear Relationships | Limited | Strong |
| Feature Interactions | Limited | Strong |
| Class Imbalance Handling | Class weights | `scale_pos_weight` |
| Test Accuracy | 82% | **92%** |
| Test Precision | 56% | **81%** |
| Test Recall | 79% | **81%** |
| Test F1 Score | 65% | **81%** |
| CV ROC-AUC | 0.871 | **0.939** |
| Explainability | Coefficients | SHAP |

Logistic Regression provides a useful interpretable benchmark, while XGBoost provides stronger overall predictive performance.

SHAP is then used to improve transparency around the more complex model.

## Model Persistence

After training and calibration, the final model is saved using Joblib.

```python
joblib.dump(calibrated_model, "credit_risk_model.pkl")
```

Saving the complete model allows it to be loaded later without retraining.

This provides a starting point for integration into applications such as:

- Credit risk dashboards
- Internal risk assessment tools
- Batch scoring pipelines
- Model APIs

## Tech Stack

### Programming

- Python

### Data Manipulation

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Explainability

- SHAP

### Data Preprocessing

- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder

### Model Selection and Validation

- Stratified Train-Test Split
- StratifiedKFold
- Cross-Validation
- RandomizedSearchCV

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Confusion Matrix
- Precision-Recall Curve
- Calibration Curve

### Probability Estimation

- CalibratedClassifierCV
- Sigmoid Calibration

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

## Skills Demonstrated

This project demonstrates practical experience with:

- End-to-end machine learning workflows
- Credit risk modelling
- Binary classification
- Imbalanced classification
- Statistical model evaluation
- Logistic Regression
- Gradient boosting
- XGBoost
- Feature preprocessing
- Scikit-learn pipelines
- Hyperparameter optimization
- Stratified cross-validation
- Probability calibration
- Explainable machine learning
- SHAP analysis
- Error analysis
- Model persistence

## What I Learned

This project reinforced that building a useful risk model involves more than maximizing accuracy.

Three areas were particularly important.

### 1. Imbalanced Classification

Default prediction requires looking beyond overall accuracy.

Precision, recall, F1 score, ROC-AUC, and Average Precision provide a more useful picture of how well the model identifies risky applicants.

### 2. Probability Quality

A classifier can rank applicants correctly while still producing probability estimates that do not accurately represent real-world risk.

Probability calibration helps address this issue.

### 3. Model Interpretability

Strong predictive performance is more useful when individual predictions can also be understood.

SHAP makes it possible to examine both overall feature influence and applicant-level predictions.

## Future Improvements

Possible extensions to the project include:

- Cost-sensitive threshold selection based on lending costs
- Additional model comparison
- More detailed probability calibration analysis
- Feature engineering based on financial ratios
- Model stability analysis across borrower segments
- Fairness and bias evaluation
- Model monitoring for changes in data distributions
- API development for real-time scoring
- Interactive dashboard for applicant-level explanations

## Conclusion

This project develops an end-to-end machine learning workflow for credit risk assessment.

A Logistic Regression model establishes an interpretable baseline, while a tuned XGBoost model improves predictive performance to approximately **92% test accuracy and 0.81 F1 score**.

The workflow goes beyond classification by incorporating:

- Stratified cross-validation
- Class imbalance handling
- Hyperparameter optimization
- Probability calibration
- SHAP explainability
- Applicant-level explanations
- False positive and false negative analysis

The final result is not only a model that predicts loan default risk, but a workflow designed to make those predictions more measurable, interpretable, and useful for credit risk analysis.

## Author

**Deepak Kushwaha**  
MSc Data Science, University of Naples Federico II  

[GitHub](https://github.com/ideepkush) | [LinkedIn](https://www.linkedin.com/in/deepak-kushwaha-75155013a/)
