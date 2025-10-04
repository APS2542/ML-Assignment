import os
import numpy as np
import pandas as pd
import pytest

try:
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/"))
except Exception:
    mlflow = None

MODEL_URI = os.getenv("MODEL_URI", "runs:/d63658dc931c43d39fb75c89f111957f/model") 
SAMPLE_PATH = os.getenv("SAMPLE_PATH", "tests/fixtures/sample_input.csv")


def _load_sample_df():
    assert os.path.exists(SAMPLE_PATH), f"missing {SAMPLE_PATH}"
    df = pd.read_csv(SAMPLE_PATH)
    assert len(df) > 0, "sample_input.csv is empty"
    return df


def _load_model_or_skip():
    import time
    for i in range(3):
        try:
            return mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ["too many 500", "server error", "502", "504"]):
                if i < 2:
                    time.sleep(2 * (i + 1))
                    continue
                pytest.skip(f"Skip due to MLflow server 5xx: {e}")
            if any(s in msg for s in ["not authenticated", "401", "not found"]):
                pytest.skip(f"Skip due to MLflow auth/registry: {e}")
            raise


@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_model_accepts_expected_input():
    df = _load_sample_df()
    model = _load_model_or_skip()
    y_pred = model.predict(df)
    assert len(y_pred) == len(df)


@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_prediction_shape_and_values():
    df = _load_sample_df()
    model = _load_model_or_skip()
    y_pred = np.asarray(model.predict(df)).reshape(-1)
    assert y_pred.shape == (len(df),)
    assert np.all(np.isin(y_pred, [0, 1, 2, 3]))
