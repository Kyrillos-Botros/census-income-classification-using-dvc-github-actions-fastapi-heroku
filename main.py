from fastapi import FastAPI
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
  path: str

@app.get("/")
async def welcome()-> str:
    return "Welcome to census income classification APP"

@app.post("/predict")
async def predict(input: PredictInput)-> List:
    # Reading testing data
    testing_dataframe = pd.read_csv(input.path)
    cat_features = testing_dataframe.select_dtypes(include=['object']).columns
    cat_features = cat_features[:-1]
    X_test, y_test, _, _ = process_data(
        testing_dataframe, categorical_features=cat_features, label="salary", training=False, encoder=model["encoder"],
        lb=model["lb"]
    )
    preds = inference(model=model["model"], X=X_test)
    return preds.tolist()