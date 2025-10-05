from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import pickle, joblib

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE  

NUMERIC_FEATURES = ["year", "km_driven", "mileage", "max_power"]

def _align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    feats = list(getattr(model, "feature_names_in_", []))
    if not feats:
        return X
    extra = [c for c in X.columns if c not in feats]
    if extra:
        X = X.drop(columns=extra)
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    return X[feats]

@lru_cache(maxsize=1)
def _load_a1():
    mpath = MODEL_DIR / "model.pkl"
    spath = MODEL_DIR / "scalar.pkl"
    with open(mpath, "rb") as f:
        model = pickle.load(f)
    with open(spath, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def status_a1() -> str:
    ok = (MODEL_DIR / "model.pkl").exists() and (MODEL_DIR / "scalar.pkl").exists()
    return "ready" if ok else "missing A1 artifacts (model.pkl/scalar.pkl)"

def predict_a1(df: pd.DataFrame):
    model, scaler = _load_a1()
    X = df.copy()
    cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    if cols:
        X.loc[:, cols] = scaler.transform(X[cols])
    X = _align_to_model_features(X, model)
    y = model.predict(X)

    return np.exp(y)
@lru_cache(maxsize=1)
def _load_a2():
    mpath = MODEL_DIR / "model_1.pkl"
    spath = MODEL_DIR / "scalar.pkl"
    model = joblib.load(mpath)
    scaler = joblib.load(spath)
    return model, scaler

def status_a2() -> str:
    ok = (MODEL_DIR / "model_1.pkl").exists() and (MODEL_DIR / "scalar.pkl").exists()
    return "ready" if ok else "missing A2 artifacts (model_1.pkl/scalar.pkl)"

def predict_a2(df: pd.DataFrame):
    model, scaler = _load_a2()
    X = df.copy()
    cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    if cols:
        X.loc[:, cols] = scaler.transform(X[cols])
    y_log = model.predict(X, True)  
    return np.exp(y_log)

