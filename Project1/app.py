from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model and preprocessing objects
model = joblib.load("titanic_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# Input data structure
class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str


# Home route
@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}


# Health check route
@app.get("/health")
def health():
    return {"status": "API is running"}


# Prediction route
@app.post("/predict")
def predict(passenger: Passenger):

    # Create DataFrame from input
    data = pd.DataFrame([{
        "pclass": passenger.pclass,
        "sex": passenger.sex,
        "age": passenger.age,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
        "fare": passenger.fare,
        "embarked": passenger.embarked
    }])

    # Convert categorical variables
    data = pd.get_dummies(
        data,
        columns=["sex", "embarked"],
        drop_first=True
    )

    # Make sure columns are in the same order as training data
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale the input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)[0]

    # Result
    if prediction == 1:
        result = "Survived"
    else:
        result = "Did not survive"

    return {
        "prediction": int(prediction),
        "result": result
    }