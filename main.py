from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("credit_risk_model.pkl")
threshold = joblib.load("best_threshold.pkl")


class CreditInput(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(data: CreditInput):

    input_df = pd.DataFrame([data.model_dump()])

    probability = model.predict_proba(input_df)[:, 1][0]

    prediction = int(probability >= threshold)

    return {
    "default_probability": float(probability),
    "prediction": prediction,
    "risk": "High Risk" if prediction == 1 else "Low Risk"
    }
 