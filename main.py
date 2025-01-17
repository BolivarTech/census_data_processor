"""
FastAPI instance and model inference

FastAPI instance and model inference

By: Julian Bolivar
Version: 1.0.0
Date:  2023-05-29
Revision 1.0.0 (2023-05-29): Initial Release
"""

# System's imports
import os
import pickle

# FastAPI imports
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
# Data managment imports
import pandas as pd

# ML imports
from src.ml.data import process_data

# path to saved artifacts
SAVEPATH = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Declare the data object with its components and their type.


class InputData(BaseModel):
    """
        Provide the FastAPI Class template
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        """
            FastApi schema example
        """

        schema_extra = {
            "example": {
                'age': 30,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 60,
                'native_country': "United-States"
            }
        }


# instantiate FastAPI app
app = FastAPI(title="Inference API",
              description="API that takes a sample and runs an inference",
              version="1.0.0")

# load model artifacts on startup of the application to reduce latency


# @app.on_event("startup")
# async def startup_event():
#     global model, encoder, lb
#     # if saved model exits, load the model from disk
#     if os.path.isfile(os.path.join(SAVEPATH, filename[0])):
#         with open(os.path.join(SAVEPATH, filename[0]), "rb") as file:
#             model = pickle.load(file)
#         with open(os.path.join(SAVEPATH, filename[1]), "rb") as file:
#             encoder = pickle.load(file)
#         with open(os.path.join(SAVEPATH, filename[2]), "rb") as file:
#             lb = pickle.load(file)


@app.get("/")
async def greetings():
    """
    Simple greeting message
    """
    return "Welcome to model API"


# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/inference/")
async def ingest_data(inference: InputData):
    """
    Process the inference API request

    parameters:
    inference (InputData): vaues from POST request

    returns:
    interence data and 'prediction' value appended at end
    """
    data = {'age': inference.age,
            'workclass': inference.workclass,
            'fnlgt': inference.fnlgt,
            'education': inference.education,
            'education-num': inference.education_num,
            'marital-status': inference.marital_status,
            'occupation': inference.occupation,
            'relationship': inference.relationship,
            'race': inference.race,
            'sex': inference.sex,
            'capital-gain': inference.capital_gain,
            'capital-loss': inference.capital_loss,
            'hours-per-week': inference.hours_per_week,
            'native-country': inference.native_country,
            }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(SAVEPATH, filename[0])):
        with open(os.path.join(SAVEPATH, filename[0]), "rb") as file:
            model = pickle.load(file)
        with open(os.path.join(SAVEPATH, filename[1]), "rb") as file:
            encoder = pickle.load(file)
        with open(os.path.join(SAVEPATH, filename[2]), "rb") as file:
            lb = pickle.load(file)

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get model prediction which is a one-dim array like [1]
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction_resp = ">50K"
    else:
        prediction_resp = "<=50K",

    data['prediction'] = prediction_resp

    return data


if __name__ == '__main__':
    print("Script NOT to be run alone")
