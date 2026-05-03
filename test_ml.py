import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics


def test_data_split():
    """
    Test that the dataset is split correctly into
    training and testing sets.

    Verifies that the training set is larger than
    the test set and the total number of rows remains
    consistent after the split.
    """
    data = pd.read_csv("data/census.csv")

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    assert len(train) > len(test)
    assert len(train) + len(test) == len(data)


def test_process_data_output():
    """
    Test that the process_data accurately transforms
    the dataset.

    Verifies that the number of rows in features(X)
    matches labels (y) and the encoder and label
    binarizer are created.
    """
    data = pd.read_csv("data/census.csv")

    cat_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship",
        "race", "sex", "native-country"
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == y.shape[0]
    assert hasattr(encoder, "transform")
    assert hasattr(lb, "transform")


def test_metrics_values():
    """
    Test that compute_model_metrics returns
    expected evaluation values.

    Verifies that the precision, recall, and
    F1 scores are correctly computed for known
    labels and predictions.
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
