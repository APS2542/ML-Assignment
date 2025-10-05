from __future__ import annotations
import os
from functools import lru_cache
import pandas as pd
import mlflow
import numpy as np

LABEL_MAP = {"0":"Budget","1":"Standard","2":"Premium","3":"Luxury"}

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

_ERR = None

@lru_cache(maxsize=1)
def _load_pyfunc():
    uri = os.getenv("MODEL_URI", "").strip()
    if not uri:
        raise RuntimeError("MODEL_URI is not set")
    return mlflow.pyfunc.load_model(uri)

def status() -> str:
    global _ERR
    if _ERR:
        return f"error: {_ERR}"
    try:
        _load_pyfunc()
        return "ready"
    except Exception as e:
        _ERR = str(e)
        return f"error: {_ERR}"

def predict(df: pd.DataFrame):
    m = _load_pyfunc()
    y = m.predict(df)
    out = []
    for v in np.atleast_1d(y):
        key = str(v)
        out.append(LABEL_MAP.get(key, v))
    return np.array(out, dtype=object)
