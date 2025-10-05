import dash
from dash import html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from app.code.model.registry import get_predictor, REGISTRY

dash.register_page(__name__, path="/prediction_model_2", name="Predict (Model A1 / A2 / A3)")

BINARY01_FEATURES = [
    "fuel_Petrol",
    "transmission_Manual",
    "seller_type_Individual",
    "seller_type_Trustmark Dealer",
    "owner_2",
    "owner_3",
    "owner_4",
]
RAW_FEATURES = ["year", "km_driven", "mileage", "max_power"] + BINARY01_FEATURES

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
                dbc.Col(
                    [
                        dbc.Label("Choose Model"),
                        dbc.RadioItems(
                            id="inp-model",
                            options=[{"label": k, "value": k} for k in REGISTRY.keys()],
                            value=next(iter(REGISTRY.keys())), 
                            inline=True,
                        ),
                    ],
                    md=12,
                ),
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
        dbc.Button("Predict", id="btn-predict", color="primary", className="mt-3"),
        html.Div(id="pred-output", className="mt-3 fw-bold text-primary"),
        html.Div(id="pred-proba", className="mt-2"),    
        html.Div(id="pred-error", className="mt-2 text-danger"),
    ]
)

layout = dbc.Container(
    [
        html.H2("Car Price/Category Prediction"),
        html.P("Fill in the details, select model, and click Predict."),
        form,
    ],
    fluid=True,
)

@callback(
    Output("pred-output", "children"),
    Output("pred-proba", "children"),
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
        X_raw = build_raw_df({
            "year": year, "km": km, "mileage": mileage, "power": power,
            "fuel": fuel, "trans_manual": trans_manual,
            "seller_individual": seller_individual, "seller_trustmark": seller_trustmark,
            "owner2": owner2, "owner3": owner3, "owner4": owner4,
        })

        predict_fn, status_fn = get_predictor(model_key)

        st = status_fn()
        if st != "ready":
            return "", "", f"Model not ready: {st}"

        y = predict_fn(X_raw)
        y0 = y[0] if hasattr(y, "__len__") and len(y) > 0 else y

        if isinstance(y0, (str, bytes)):
            msg = f"Predicted class: {y0}"
            return msg, "", ""
        else:
            try:
                val = float(y0)
                msg = f"Estimated selling price ≈ {val:,.0f}"
                return msg, "", ""
            except Exception:
                return f"Prediction: {y0}", "", ""

    except Exception as e:
        return "", "", f"Prediction failed: {e}"


