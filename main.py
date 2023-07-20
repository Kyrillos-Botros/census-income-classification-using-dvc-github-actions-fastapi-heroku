from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import pandas as pd
import pickle
from starter.ml.data import process_data
from starter.ml.model import inference


with open('model/rfmodel.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


class PredictInput(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    education: str
    education_num: int
    fnlgt: int
    hours_per_week: int
    marital_status: str
    native_country: str
    occupation: str
    race: str
    relationship: str
    sex: str
    workclass: str
    class Config:
        schema_extra = {
            "examples": [
                {
                    "age": 30,
                    "capital_gain": 3000,
                    "capital_loss": 0,
                    "education": "Bachelors",
                    "education_num": 15,
                    "fnlgt": 2443,
                    "hours_per_week": 50,
                    "marital_status":"Divorced",
                    "native_country": "Cuba",
                    "occupation": "Adm-clerical",
                    "race": "Black",
                    "relationship": "Wife",
                    "sex": "Female",
                    "workclass": "State-gov"
                }
            ]
        }


@app.get("/")
async def welcome() -> str:
    return "Welcome to census income classification APP"


@app.post("/predict")
async def predict(input: PredictInput):
    # Reading testing data
    print(input)
    testing_dataframe = pd.DataFrame([input.dict(by_alias=True)], index=[0])
    cat_features = testing_dataframe.select_dtypes(include=['object']).columns
    cat_features = cat_features
    X_test, y_test, _, _ = process_data(
        testing_dataframe,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=model["encoder"],
        lb=model["lb"]
    )
    preds = inference(model=model["model"], X=X_test)

    return preds.tolist()

