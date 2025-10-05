import os
import pytest
import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.pyfunc  
except Exception:
    mlflow = None

ALLOWED_LABELS = {"Budget", "Standard", "Premium", "Luxury"}

MODEL_URI = os.getenv("MODEL_URI", "").strip()

def _sample_df() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_input.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    return pd.DataFrame([{
        "year": 2018,
        "km_driven": 50000,
        "mileage": 18,
        "max_power": 82,
        "fuel_Petrol": 1,
        "transmission_Manual": 1,
        "seller_type_Individual": 1,
        "seller_type_Trustmark Dealer": 0,
        "owner_2": 0,
        "owner_3": 0,
        "owner_4": 0,
    }])

@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_model_accepts_expected_input():
    assert MODEL_URI, "MODEL_URI env is empty for CI"
    df = _sample_df()
    model = mlflow.pyfunc.load_model(MODEL_URI)
    pred = model.predict(df)
    assert np.asarray(pred).shape[0] == 1

@pytest.mark.skipif(mlflow is None, reason="mlflow not installed")
def test_prediction_shape_and_values():
    assert MODEL_URI, "MODEL_URI env is empty for CI"
    df = _sample_df()
    model = mlflow.pyfunc.load_model(MODEL_URI)
    y = model.predict(df)
    val = str(y[0])
    assert val in ALLOWED_LABELS, f"unexpected label: {val}"
