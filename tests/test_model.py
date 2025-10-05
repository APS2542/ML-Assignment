import os
import numpy as np
import pytest
import mlflow.sklearn

def _resolve_uri():
    uri = os.getenv("TEST_MODEL_URI", "").strip()
    if not uri:
        uri = os.getenv("MODEL_URI", "").strip()
    if not uri:
        name = os.getenv("MODEL_NAME", "").strip()
        if name:
            uri = f"models:/{name}/Staging"
    return uri

@pytest.fixture(scope="session")
def skl_model():
    uri = _resolve_uri()
    if not uri:
        pytest.skip("No MODEL URI provided via env; skip model tests.")
    return mlflow.sklearn.load_model(uri)

def test_model_has_predict(skl_model):
    assert hasattr(skl_model, "predict")

def test_model_output_shape(skl_model):
    n = getattr(skl_model, "n_features_in_", None)
    assert n is not None and n > 0
    X = np.zeros((2, n))
    y = skl_model.predict(X)
    assert len(y) == 2
