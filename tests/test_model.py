import os
import numpy as np
import pandas as pd
import pytest

try:
    import mlflow
except Exception:
    mlflow = None

MODEL_URI = os.getenv("MODEL_URI", "models:/st126130-a3-model/Staging")
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "tests/fixtures/sample_input.csv")


def _load_sample_df():
    assert os.path.exists(SAMPLE_PATH), f"missing sample CSV at {SAMPLE_PATH}"
    df = pd.read_csv(SAMPLE_PATH)
    assert len(df) > 0, "sample_input.csv is empty"
    return df


@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_model_accepts_expected_input():
    df = _load_sample_df()
    model = mlflow.pyfunc.load_model(MODEL_URI)
    y_pred = model.predict(df)
    assert len(y_pred) == len(df)


@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_prediction_shape_and_values():
    df = _load_sample_df()
    model = mlflow.pyfunc.load_model(MODEL_URI)
    y_pred = np.asarray(model.predict(df)).reshape(-1)
    assert y_pred.shape == (len(df),)
    assert np.all(np.isin(y_pred, [0, 1, 2, 3]))
