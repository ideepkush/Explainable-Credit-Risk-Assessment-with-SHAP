# Explainable Credit Risk Assessment with SHAP

End-to-end credit risk modelling project covering data validation, imbalanced classification, model tuning, probability calibration, SHAP explainability, REST API development, Docker containerization, and Kubernetes deployment.

The project compares Logistic Regression with XGBoost for loan default prediction, explains individual credit decisions using SHAP, and serves the trained model through a FastAPI application deployed with Docker and Kubernetes.

## Project Highlights

- Built an end-to-end credit risk classification pipeline using Scikit-learn and XGBoost
- Compared an interpretable Logistic Regression baseline with a tuned XGBoost model
- Achieved approximately **92% test accuracy and 0.81 F1 score** with XGBoost
- Achieved approximately **0.939 ROC-AUC** in 5-fold stratified cross-validation
- Handled class imbalance using class weighting and `scale_pos_weight`
- Tuned XGBoost using `RandomizedSearchCV` with 150 parameter combinations
- Calibrated predicted probabilities using `CalibratedClassifierCV`
- Used SHAP for global and applicant-level explainability
- Analysed false positives and false negatives from a lending perspective
- Served the trained model through a FastAPI REST API
- Containerized the application with Docker
- Published the Docker image to Docker Hub
- Deployed the API locally on Kubernetes with two replicas and a Kubernetes Service

## Tech Stack

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn, XGBoost  
**Explainability:** SHAP  
**API:** FastAPI, Pydantic, Uvicorn  
**Deployment:** Docker, Docker Hub, Kubernetes  
**Visualization:** Matplotlib, Seaborn  
**Model Persistence:** Joblib  

---

## Business Problem

Credit risk models help lenders estimate whether a borrower is likely to default.

The problem involves balancing two types of risk:

- approving a high-risk borrower can lead to financial loss
- rejecting a reliable borrower can lead to lost business

For this reason, a useful credit risk model should do more than produce a binary prediction.

It should also help answer:

- How likely is this applicant to default?
- Which factors are influencing the prediction?
- How reliable is the predicted probability?
- How many risky applicants are being missed?
- How many reliable applicants are being incorrectly classified as risky?

This project treats credit risk as both a prediction problem and an explainability problem.

---

## Key Results

### Held-Out Test Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.82 | 0.56 | 0.79 | 0.65 |
| XGBoost | **0.92** | **0.81** | **0.81** | **0.81** |

### 5-Fold Stratified Cross-Validation

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.871 | 0.812 | 0.545 | 0.778 | 0.641 |
| XGBoost | **0.939** | **0.909** | **0.791** | **0.786** | **0.788** |

The tuned XGBoost model also achieved an average precision score of approximately **0.90** during hyperparameter search.

The results show that XGBoost provides a stronger balance between identifying default cases and limiting false alarms.

---

## Dataset

The dataset contains **32,581 loan applications**.

The target variable is:

```text
loan_status
```

where:

```text
0 = non-default
1 = default
```

### Class Distribution

```text
Non-default: 25,473
Default:      7,108
```

The dataset is therefore imbalanced, with default cases representing the minority class.

### Features

The model uses information including:

- applicant age
- annual income
- employment length
- home ownership
- loan purpose
- loan grade
- loan amount
- interest rate
- loan-to-income ratio
- previous default history
- credit history length

These features capture both borrower characteristics and loan characteristics.

---

## End-to-End Workflow

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
XGBoost
        |
        v
Cross-Validation
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
        |
        v
FastAPI REST API
        |
        v
Docker
        |
        v
Docker Hub
        |
        v
Kubernetes Deployment
```

---

## Data Validation and Cleaning

Before model training, the dataset is checked for data quality issues that could affect the results.

The validation process includes:

- missing value inspection
- duplicate detection
- unrealistic applicant ages
- employment length validation
- employment history relative to age
- extreme numerical observations
- loan amount validation
- interest rate inspection
- target class distribution analysis

The purpose of this stage is to prevent unrealistic or inconsistent observations from influencing model training.

---

## Exploratory Data Analysis

Exploratory analysis is used to understand how borrower and loan characteristics differ between default and non-default cases.

The analysis focuses on variables such as:

- income
- loan amount
- interest rate
- loan-to-income ratio
- loan grade
- loan purpose
- home ownership
- previous default history

The goal is to identify patterns that may contain useful information for credit risk prediction.

---

## Train-Test Strategy

The cleaned dataset is divided into training and test sets using a stratified split.

Stratification preserves the proportion of default and non-default observations across both datasets.

This is particularly important because the target variable is imbalanced.

The test set is kept separate from model training and hyperparameter selection so that it can provide an independent evaluation of final model performance.

---

## Preprocessing

Preprocessing is implemented using Scikit-learn `Pipeline` and `ColumnTransformer`.

Keeping preprocessing inside the pipeline ensures that the same transformations are applied during both model training and inference.

### Numerical Features

Numerical variables are processed using:

- median imputation
- standard scaling where required

### Categorical Features

Categorical variables are processed using:

- missing value imputation
- one-hot encoding

Different preprocessing configurations are used for Logistic Regression and XGBoost because the models have different requirements.

---

## Logistic Regression Baseline

Logistic Regression is used as an interpretable baseline.

The baseline pipeline includes:

- median imputation
- numerical standardization
- one-hot encoding
- balanced class weights

### Test Performance

```text
Accuracy:  82%
Precision: 56%
Recall:    79%
F1 Score:  65%
```

The relatively high recall means the model identifies many default cases.

However, the lower precision means more reliable borrowers are also classified as risky.

This provides a useful benchmark for evaluating whether a more complex model delivers meaningful improvement.

---

## Handling Class Imbalance

Loan defaults represent a smaller proportion of the dataset.

A model trained without considering this imbalance could obtain reasonable accuracy while favouring the majority non-default class.

Different strategies are therefore used for the two models.

### Logistic Regression

Balanced class weights give more importance to errors on the minority default class.

### XGBoost

XGBoost uses `scale_pos_weight`.

The approximate value used is:

```text
3.63
```

This increases the importance of default observations during training.

---

## XGBoost

XGBoost is used as the main nonlinear model.

Compared with Logistic Regression, gradient boosted trees can capture:

- nonlinear relationships
- interactions between features
- more complex decision boundaries

These properties are useful in credit risk because borrower behaviour is unlikely to depend on completely linear relationships.

---

## Hyperparameter Tuning

The XGBoost model is optimized using `RandomizedSearchCV`.

The search evaluates **150 parameter combinations** using **5-fold stratified cross-validation**.

The optimization metric is:

```text
Average Precision
```

Average Precision is useful for this problem because the default class is the minority class.

The search explores parameters including:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`

The objective is to improve predictive performance while controlling model complexity.

---

## Cross-Validation

A single train-test split can depend heavily on one particular division of the data.

To obtain a more reliable estimate of generalization performance, both models are evaluated using 5-fold Stratified Cross-Validation.

The evaluation includes:

- ROC-AUC
- accuracy
- precision
- recall
- F1 score

### Cross-Validation Results

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.871 | **0.939** |
| Accuracy | 0.812 | **0.909** |
| Precision | 0.545 | **0.791** |
| Recall | 0.778 | **0.786** |
| F1 Score | 0.641 | **0.788** |

XGBoost provides the strongest overall balance across the evaluation metrics.

---

## Why Accuracy Alone Is Not Enough

Accuracy is not sufficient for evaluating this problem because most borrowers belong to the non-default class.

Several additional metrics are therefore considered.

### Precision

Among borrowers predicted to default, how many actually default?

Higher precision reduces the number of reliable borrowers incorrectly classified as risky.

### Recall

Among borrowers who actually default, how many are identified by the model?

Higher recall reduces the number of risky borrowers that the model fails to detect.

### F1 Score

F1 combines precision and recall into a single metric.

### ROC-AUC

ROC-AUC measures how well the model ranks risky borrowers above lower-risk borrowers across different classification thresholds.

### Average Precision

Average Precision evaluates performance across the precision-recall curve and is particularly useful for imbalanced classification.

---

## Probability Calibration

Credit risk decisions often depend on predicted probabilities rather than only binary classifications.

For example:

```text
Applicant A: 12% probability of default
Applicant B: 67% probability of default
```

These probabilities can be used for risk ranking and decision support.

However, a model can classify borrowers correctly while still producing poorly calibrated probabilities.

A predicted probability of 30% should ideally correspond to a group where approximately 30% of borrowers actually default.

The XGBoost probabilities are therefore calibrated using:

```text
CalibratedClassifierCV
```

with sigmoid calibration and five-fold cross-validation.

Calibration curves are used to compare the original and calibrated probabilities.

---

## Explainable AI with SHAP

Strong predictive performance is not sufficient for a credit risk model.

It is also important to understand why the model considers an applicant risky.

SHAP is used to explain the XGBoost model at both global and individual levels.

### Global Explainability

SHAP summary plots help answer:

- Which features have the greatest overall influence?
- Which values increase predicted risk?
- Which values reduce predicted risk?
- How does the model use borrower and loan information?

This provides a global view of the model's decision process.

### Applicant-Level Explainability

SHAP waterfall plots are used to explain individual predictions.

For each applicant:

```text
Model baseline
      +
Applicant-specific feature contributions
      =
Final model prediction
```

Some characteristics push predicted risk higher while others reduce it.

This makes individual model decisions easier to interpret.

---

## Error Analysis

The tuned XGBoost model produced:

```text
False Positives: 258
False Negatives: 256
```

These errors have different business consequences.

### False Positive

The model predicts default, but the borrower does not actually default.

Possible consequences include:

- rejecting a reliable borrower
- losing a potential customer
- reducing lending opportunities
- losing potential revenue

### False Negative

The model predicts non-default, but the borrower later defaults.

Possible consequences include:

- credit losses
- higher portfolio risk
- collection costs
- additional risk management activity

Analysing these errors helps connect technical model performance with lending decisions.

---

## Model Comparison

| Area | Logistic Regression | XGBoost |
|---|---|---|
| Interpretability | High | Lower without explainability tools |
| Nonlinear Relationships | Limited | Strong |
| Feature Interactions | Limited | Strong |
| Class Imbalance Handling | Class weights | `scale_pos_weight` |
| Test Accuracy | 82% | **92%** |
| Test Precision | 56% | **81%** |
| Test Recall | 79% | **81%** |
| Test F1 Score | 65% | **81%** |
| CV ROC-AUC | 0.871 | **0.939** |
| Explainability | Coefficients | SHAP |

Logistic Regression provides a useful interpretable benchmark.

XGBoost provides stronger predictive performance, while SHAP is used to improve transparency around the more complex model.

---

# Model Deployment

## Model Persistence

After training and calibration, the final model is persisted using Joblib.

```python
joblib.dump(calibrated_model, "credit_risk_model.pkl")
```

The selected decision threshold is also stored separately.

```python
joblib.dump(best_threshold, "best_threshold.pkl")
```

This allows the model and threshold to be reused without retraining.

The saved artifacts are then loaded by the FastAPI application during startup.

---

## FastAPI Model Serving

The trained model is exposed through a REST API using FastAPI.

The application loads:

```text
credit_risk_model.pkl
best_threshold.pkl
```

at startup.

The API accepts borrower information, applies the same preprocessing pipeline used during training, calculates the probability of default, and applies the saved classification threshold.

### API Endpoints

#### Health Check

```text
GET /
```

Example response:

```json
{
  "message": "Credit Risk API is running"
}
```

#### Prediction

```text
POST /predict
```

Example request:

```json
{
  "person_age": 32,
  "person_income": 45000,
  "person_home_ownership": "RENT",
  "person_emp_length": 2,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 15000,
  "loan_int_rate": 13.5,
  "loan_percent_income": 0.33,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 5
}
```

The API returns:

```text
default_probability
prediction
risk
```

### Prediction Flow

```text
Client Request
      |
      v
FastAPI
      |
      v
Pydantic Validation
      |
      v
Pandas DataFrame
      |
      v
Saved Preprocessing Pipeline
      |
      v
Calibrated XGBoost Model
      |
      v
Default Probability
      |
      v
Decision Threshold
      |
      v
Risk Classification
```

FastAPI automatically provides Swagger documentation.

Run the API locally:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

The FastAPI application is containerized using Docker.

Containerization packages the application and its dependencies into a consistent runtime environment.

The Docker image contains:

- Python
- FastAPI
- Uvicorn
- Pandas
- Joblib
- Scikit-learn
- XGBoost
- application code
- model artifacts required for inference

### Build

```bash
docker build -t credit-risk-api .
```

### Run

```bash
docker run -p 8000:8000 credit-risk-api
```

The API can then be accessed at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Dependency Reproducibility

Model persistence requires the serving environment to remain compatible with the environment used during training.

The persisted model was created with:

```text
scikit-learn==1.7.2
```

The same version is pinned in `requirements.txt`.

This avoids incompatibility when loading serialized Scikit-learn objects.

---

## Docker Hub

After validating the container locally, the Docker image is published to Docker Hub.

Image:

```text
deepakkushwahaa/credit-risk-api:latest
```

Pull the image:

```bash
docker pull deepakkushwahaa/credit-risk-api:latest
```

Run it:

```bash
docker run -p 8000:8000 deepakkushwahaa/credit-risk-api:latest
```

Publishing the image to a registry allows the same application image to be reused across environments.

---

## Kubernetes

The containerized API is deployed locally using Kubernetes through Docker Desktop.

The deployment uses:

```text
deployment.yaml
service.yaml
```

### Deployment

`deployment.yaml` defines the application workload.

The deployment uses:

```text
replicas: 2
```

This tells Kubernetes to maintain two running instances of the API.

```text
Kubernetes Deployment
        |
        +------------------+
        |                  |
        v                  v
      Pod 1              Pod 2
```

Deploy:

```bash
kubectl apply -f deployment.yaml
```

Check the pods:

```bash
kubectl get pods
```

Both pods should show:

```text
READY   STATUS
1/1     Running
```

---

## Kubernetes Service

Kubernetes pods can be recreated and their internal addresses can change.

A Kubernetes Service provides a stable way to access the application.

The service selects pods using:

```text
app: credit-risk-api
```

and forwards traffic to port:

```text
8000
```

Architecture:

```text
Client
   |
   v
Kubernetes Service
   |
   +-------------------+
   |                   |
   v                   v
Pod 1                 Pod 2
FastAPI               FastAPI
```

Create the service:

```bash
kubectl apply -f service.yaml
```

Check it:

```bash
kubectl get service credit-risk-service
```

---

## Local Kubernetes Access

For local testing, the Kubernetes service can be exposed using port forwarding.

```bash
kubectl port-forward service/credit-risk-service 8000:8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

If port `8000` is already occupied:

```bash
kubectl port-forward service/credit-risk-service 8001:8000
```

Then open:

```text
http://127.0.0.1:8001/docs
```

---

## Deployment Architecture

```text
                   Credit Risk Dataset
                           |
                           v
                Data Validation & Cleaning
                           |
                           v
                     Preprocessing
                           |
                           v
                    Model Training
                           |
                           v
                 Hyperparameter Tuning
                           |
                           v
                Probability Calibration
                           |
                           v
                    SHAP Analysis
                           |
                           v
                   Model Persistence
                           |
                           v
                    FastAPI REST API
                           |
                           v
                       Docker
                           |
                           v
                     Docker Hub
                           |
                           v
                Kubernetes Deployment
                           |
                  +--------+--------+
                  |                 |
                  v                 v
                Pod 1             Pod 2
                  |                 |
                  +--------+--------+
                           |
                           v
                  Kubernetes Service
                           |
                           v
                       API Client
```

This deployment pipeline demonstrates how a machine learning model can move from experimentation in a notebook to a reusable prediction service.

---

## Project Structure

```text
Explainable-Credit-Risk-Assessment-with-SHAP/
│
├── Credit_Risk.ipynb
├── main.py
├── Dockerfile
├── requirements.txt
├── deployment.yaml
├── service.yaml
├── .gitignore
└── README.md
```

### `Credit_Risk.ipynb`

Contains data analysis, preprocessing, model training, validation, calibration, SHAP analysis, and model persistence.

### `main.py`

Contains the FastAPI application and prediction endpoint.

### `Dockerfile`

Defines the Docker image used to run the API.

### `requirements.txt`

Contains the Python dependencies required by the application.

### `deployment.yaml`

Defines the Kubernetes Deployment and application replicas.

### `service.yaml`

Defines the Kubernetes Service used to access the application pods.

---

## Technical Skills Demonstrated

### Machine Learning

- Logistic Regression
- XGBoost
- binary classification
- imbalanced classification
- probability estimation
- probability calibration

### Data Processing

- Pandas
- NumPy
- Scikit-learn Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder

### Model Selection

- stratified train-test split
- StratifiedKFold
- cross-validation
- RandomizedSearchCV
- hyperparameter optimization

### Evaluation

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- Average Precision
- confusion matrix
- precision-recall analysis
- calibration curves

### Explainability

- SHAP
- global feature importance
- applicant-level explanations
- waterfall plots

### API Development

- FastAPI
- Pydantic
- Uvicorn
- REST API development
- input validation
- model serving

### Deployment

- Docker
- Docker Hub
- Kubernetes
- Kubernetes Deployments
- Kubernetes Pods
- replicas
- Kubernetes Services
- port forwarding
- dependency reproducibility

---

## What I Learned

### 1. Imbalanced Classification

Accuracy alone is not enough when the target classes are imbalanced.

Precision, recall, F1, ROC-AUC, and Average Precision provide a more complete view of model performance.

### 2. Probability Quality

A classifier can rank borrowers correctly while still producing probability estimates that do not accurately represent real-world risk.

Probability calibration helps make those estimates more meaningful.

### 3. Model Interpretability

Strong model performance is more useful when individual predictions can also be understood.

SHAP makes it possible to inspect both global model behaviour and individual borrower predictions.

### 4. Deployment and Reproducibility

Training a model is only one part of building a complete machine learning system.

The deployment process showed the importance of keeping the training and serving environments consistent.

Version differences between Scikit-learn environments can cause persisted models to fail during inference.

Docker provides a reproducible runtime environment, while Kubernetes provides a way to manage multiple application instances.

---

## Future Improvements

Possible extensions include:

- cost-sensitive threshold selection based on lending costs
- further validation of the classification threshold
- additional model comparison
- more detailed calibration analysis
- financial ratio feature engineering
- borrower segment stability analysis
- fairness and bias evaluation
- cloud deployment
- model monitoring
- data drift detection
- prediction logging
- API monitoring
- automated tests
- CI/CD pipeline
- automated Docker image builds
- Kubernetes readiness and liveness probes
- Kubernetes CPU and memory limits
- interactive dashboard for applicant-level explanations

---

## Conclusion

This project demonstrates an end-to-end credit risk machine learning workflow.

The modelling stage covers data validation, preprocessing, class imbalance handling, Logistic Regression, XGBoost, stratified cross-validation, hyperparameter tuning, probability calibration, SHAP explainability, and error analysis.

The tuned XGBoost model achieved approximately **92% test accuracy, 81% precision, 81% recall, and 0.81 F1 score**, with a cross-validation ROC-AUC of approximately **0.939**.

The project also goes beyond notebook-based model development.

The trained model is served through a FastAPI REST API, containerized with Docker, published to Docker Hub, and deployed on Kubernetes with two application replicas and a Kubernetes Service.

The result is a practical example of moving a machine learning model from experimentation to a reproducible and deployable application.

---

## Author

**Deepak Kushwaha**  
MSc Data Science  
University of Naples Federico II

[GitHub](https://github.com/ideepkush) | [LinkedIn](https://www.linkedin.com/in/deepak-kushwaha-75155013a/)