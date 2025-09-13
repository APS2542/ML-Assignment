import dash
from dash import html, callback, Output, Input, State
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import mlflow.sklearn
from model.model import *
import joblib
import os 
import mlflow.sklearn

import dash
from dash import html, callback, Output, Input, State
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from functools import lru_cache

dash.register_page(__name__, path="/prediction_model_2", name="Predict (Model 1 / 2)")

# Paths & Configs
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"

MODEL_CONFIG = {
    # Model 1 
    "model_1": {
        "model_path": MODEL_DIR / "model.pkl",     # pickle
        "scalar_path": MODEL_DIR / "scalar.pkl",   # pickle
        "loader": "pickle",                        # "pickle" | "joblib"
        "numeric_features": ['year',"km_driven", "mileage", "max_power"],
        "output_transform": "exp",                 # "exp" | "identity"
    },
    # Model 2
    "model_2": {
        "model_path": MODEL_DIR / "model_1.pkl",   # joblib
        "scalar_path": MODEL_DIR / "scalar.pkl",   # joblib
        "loader": "joblib",
        "numeric_features": ["year", "km_driven", "mileage", "max_power"],
        "output_transform": "identity",
    },
}

BINARY01_FEATURES = [
    "fuel_Petrol",
    "transmission_Manual",
    "seller_type_Individual",
    "seller_type_Trustmark Dealer",
    "owner_2",
    "owner_3",
    "owner_4",]

RAW_FEATURES = ["year", "km_driven", "mileage", "max_power"] + BINARY01_FEATURES

# Loaders (cached)
@lru_cache(maxsize=4)
def load_artifacts(model_key: str):
    cfg = MODEL_CONFIG[model_key]
    model_path = cfg["model_path"]
    scalar_path = cfg["scalar_path"]
    loader = cfg["loader"]

    if loader == "pickle":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scalar_path, "rb") as f:
            scalar = pickle.load(f)
    elif loader == "joblib":
        model = joblib.load(model_path)
        scalar = joblib.load(scalar_path)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    
    try:
        model_features = list(getattr(model, "feature_names_in_", []))
    except Exception:
        model_features = []

    return model, scalar, model_features


# Helpers
def to01(x):
    try:
        return 1 if int(x) == 1 else 0
    except Exception:
        return 0

def build_raw_df(values: dict) -> pd.DataFrame:
    row = {
        "year": values["year"],
        "km_driven": values["km"],
        "mileage": values["mileage"],
        "max_power": values["power"],
        "fuel_Petrol": to01(values["fuel"]),
        "transmission_Manual": to01(values["trans_manual"]),
        "seller_type_Individual": to01(values["seller_individual"]),
        "seller_type_Trustmark Dealer": to01(values["seller_trustmark"]),
        "owner_2": to01(values["owner2"]),
        "owner_3": to01(values["owner3"]),
        "owner_4": to01(values["owner4"]),
    }
    return pd.DataFrame([row], columns=RAW_FEATURES)


# UI
def num01_input(id_, label):
    return dbc.Col(
        [
            dbc.Label(f"{label} (1=True, 0=False)", html_for=id_),
            dbc.Input(id=id_, type="number", min=0, max=1, step=1, placeholder="Enter 0 or 1"),
        ],
        md=4,
    )

form = dbc.Form(
    [
        dbc.Row(
            [
                dbc.Col([dbc.Label("Choose Model"),
                         dbc.RadioItems(
                             id="inp-model",
                             options=[{"label":"Model 1: RandomForestRegressor","value":"model_1"},
                                      {"label":"Model 2: Polynomial","value":"model_2"}],
                             value="model_1",
                             inline=True,
                         )], md=12),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col([dbc.Label("Year"), dbc.Input(id="inp-year", type="number", placeholder="e.g., 2020")], md=4),
                dbc.Col([dbc.Label("km_driven"), dbc.Input(id="inp-km", type="number", placeholder="Total km driven")], md=4),
                dbc.Col([dbc.Label("mileage"), dbc.Input(id="inp-mileage", type="number", placeholder="e.g., 18")], md=4),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col([dbc.Label("max_power (bhp)"), dbc.Input(id="inp-power", type="number", placeholder="e.g., 82")], md=4),
                num01_input("inp-fuel-petrol", "fuel_Petrol"),
                num01_input("inp-trans-manual", "transmission_Manual"),
            ],
            className="g-2 mt-1",
        ),
        dbc.Row(
            [
                num01_input("inp-seller-individual", "seller_type_Individual"),
                num01_input("inp-seller-trustmark", "seller_type_Trustmark Dealer"),
                num01_input("inp-owner-2", "owner_2"),
            ],
            className="g-2 mt-1",
        ),
        dbc.Row(
            [
                num01_input("inp-owner-3", "owner_3"),
                num01_input("inp-owner-4", "owner_4"),
            ],
            className="g-2 mt-1",
        ),
        dbc.Button("Predict Price", id="btn-predict", color="primary", className="mt-3"),
        html.Div(id="pred-output", className="mt-3 fw-bold text-primary"),
        html.Div(id="pred-error", className="mt-2 text-danger"),
    ]
)

layout = dbc.Container(
    [
        html.H2("Car Price Prediction"),
        html.P("Fill in the details, select model, and click Predict."),
        form,
    ],
    fluid=True,
)


# Callback part
@callback(
    Output("pred-output", "children"),
    Output("pred-error", "children"),
    Input("btn-predict", "n_clicks"),
    State("inp-model", "value"),
    State("inp-year", "value"),
    State("inp-km", "value"),
    State("inp-mileage", "value"),
    State("inp-power", "value"),
    State("inp-fuel-petrol", "value"),
    State("inp-trans-manual", "value"),
    State("inp-seller-individual", "value"),
    State("inp-seller-trustmark", "value"),
    State("inp-owner-2", "value"),
    State("inp-owner-3", "value"),
    State("inp-owner-4", "value"),
    prevent_initial_call=True,
)
def do_predict(n, model_key, year, km, mileage, power, fuel, trans_manual,
               seller_individual, seller_trustmark, owner2, owner3, owner4):
    try:
        # 1) Build raw df
        X_raw = build_raw_df({
            "year": year, "km": km, "mileage": mileage, "power": power,
            "fuel": fuel, "trans_manual": trans_manual,
            "seller_individual": seller_individual, "seller_trustmark": seller_trustmark,
            "owner2": owner2, "owner3": owner3, "owner4": owner4,
        })

        # 2) Load artifacts
        model, scalar, model_features = load_artifacts(model_key)
        cfg = MODEL_CONFIG[model_key]

        # 3) Scale only configured numeric columns (that actually exist)
        num_cols = [c for c in cfg["numeric_features"] if c in X_raw.columns]
        if num_cols:
            X_raw.loc[:, num_cols] = scalar.transform(X_raw[num_cols])

        # 4) Align columns to model
        if model_features:
            # drop extras, add missing=0, then reorder
            extra = [c for c in X_raw.columns if c not in model_features]
            if extra:
                X_raw = X_raw.drop(columns=extra)
            for c in model_features:
                if c not in X_raw.columns:
                    X_raw[c] = 0
            X = X_raw[model_features]
        else:
            X = X_raw  # fallback

        # 5) Predict
        if model_key == "model_2":           # Poly model returns log(price)
            y_pred = model.predict(X, True)
            pred = float(np.exp(y_pred[0]))
        else:                                 # Model 1 
            y_pred = model.predict(X)
            pred = float(np.exp(y_pred[0]))

        # 6) Return TWO outputs (message, error="")
        return f"Estimated selling price ≈ {pred:,.0f}", ""
    except Exception as e:
        # Return TWO outputs (empty message, error text)
        return "", f"Prediction failed: {e}"


