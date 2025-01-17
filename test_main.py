"""
Test main API

Tets the main API functionabilities

By: Julian Bolivar
Version: 1.0.0
Date:  2023-05-29
Revision 1.0.0 (2023-05-29): Initial Release
"""

# Import FastApi App
from main import app

from fastapi.testclient import TestClient
import json
import logging

client = TestClient(app)


def test_root():
    """
    Test welcome message for get at root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to model API"


def test_inference():
    """
    Test model inference output
    """
    sample = {'age': 50,
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
              'hours_per_week': 40,
              'native_country': "United-States"
              }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data)

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 50
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"] == '>50K'


def test_inference_class0():
    """
    Test model inference output for class 0
    """
    sample = {'age': 30,
              'workclass': "Private",
              'fnlgt': 234721,
              'education': "HS-grad",
              'education_num': 1,
              'marital_status': "Separated",
              'occupation': "Handlers-cleaners",
              'relationship': "Not-in-family",
              'race': "Black",
              'sex': "Male",
              'capital_gain': 0,
              'capital_loss': 0,
              'hours_per_week': 35,
              'native_country': "United-States"
              }

    data = json.dumps(sample)

    r = client.post("/inference/", data=data)

    # test response and output
    assert r.status_code == 200
    assert r.json()["age"] == 30
    assert r.json()["fnlgt"] == 234721

    # test prediction vs expected label
    logging.info(f'********* prediction = {r.json()["prediction"]} ********')
    assert r.json()["prediction"][0] == '<=50K'


def test_wrong_inference_query():
    """
    Test incomplete sample does not generate prediction
    """
    sample = {'age': 50,
              'workclass': "Private",
              'fnlgt': 234721,
              }

    data = json.dumps(sample)
    r = client.post("/inference/", data=data)

    assert 'prediction' not in r.json().keys()
    logging.warning(
        f"The sample has {len(sample)} features. Must be 14 features")


if '__name__' == '__main__':
    # Initialize logging
    logging.basicConfig(filename='tests.log',
                        level=logging.WARNING,
                        filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s')
    test_root()
    test_inference()
    test_inference_class0()
    test_wrong_inference_query()
