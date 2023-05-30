"""
Test set for the model

Test the model using the sample data

By: Julian Bolivar
Version: 1.0.0
Date:  2023-05-29
Revision 1.0.0 (2023-05-29): Initial Release
"""

# System Imports
import os
import logging
import pickle
import pytest

# Data Management import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# Models Imports
from src.ml.model import inference
from src.ml.model import compute_model_metrics
from src.ml.model import compute_confusion_matrix
from src.ml.data import process_data


@pytest.fixture(scope="module")
def data():
    """
    code to load in the data.

    returns:
    Panda's Data Frame
    """
    datapath = "./data/census_clean.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    """
    Returns data's path
    """
    return "./data/census_clean.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, _ = train_test_split(data,
                                test_size=0.20,
                                random_state=10,
                                stratify=data['salary']
                                )
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=features,
        label="salary",
        training=True
    )
    return x_train, y_train


def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    try:
        df = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't " +
            "appear to have rows and columns")
        raise err


def test_features(data, features):
    """
    Check that categorical features are in dataset
    """
    try:
        assert sorted(set(data.columns).intersection(
            features)) == sorted(features)
    except AssertionError as err:
        logging.error(
            "Testing dataset: Features are missing in the data columns")
        raise err


def test_is_model():
    """
    Check saved model is present
    """
    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        try:
            with open(savepath, 'rb') as file:
                _ = pickle.load(file)
        except Exception as err:
            logging.error(
                "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pass


def test_is_fitted_model(train_dataset):
    """
    Check saved model is fitted
    """

    x_train, _ = train_dataset
    savepath = "./model/trained_model.pkl"
    with open(savepath, 'rb') as file:
        model = pickle.load(file)

        try:
            model.predict(x_train)
        except NotFittedError as err:
            logging.error(f"Model is not fit, error {err}")
            raise err


def test_inference(train_dataset):
    """
    Check inference function
    """
    x_train, _ = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            model = pickle.load(file)

            try:
                _ = inference(model, x_train)
            except Exception as err:
                logging.error(
                    "Inference cannot be performed on"
                    + "saved model and train data")
                raise err
    else:
        pass


def test_compute_model_metrics(train_dataset):
    """
    Check calculation of performance metrics function
    """
    x_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            model = pickle.load(file)
            preds = inference(model, x_train)

            try:
                _ = compute_model_metrics(y_train, preds)
            except Exception as err:
                logging.error(
                    "Performance metrics cannot be calculated on train data")
                raise err
    else:
        pass


def test_compute_confusion_matrix(train_dataset):
    """
    Check calculation of confusion matrix function
    """
    x_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            model = pickle.load(file)
            preds = inference(model, x_train)

            try:
                _ = compute_confusion_matrix(y_train, preds)
            except Exception as err:
                logging.error(
                    "Confusion matrix cannot be calculated on train data")
                raise err
    else:
        pass
