import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    [
        html.H2("Welcome to Car Price Prediction App (version 3.0)"),
        html.P("This app predicts the estimated selling price or class of a car."),

        html.H3("Features used in the model:"),
        html.Ul(
            [
                html.Li("year (manufacturing year)"),
                html.Li("km_driven (kilometers driven)"),
                html.Li("mileage (fuel efficiency, km/l)"),
                html.Li("max_power (engine power in bhp)"),
                html.Li("fuel_Petrol"),
                html.Li("transmission_Manual"),
                html.Li("seller_type_Individual"),
                html.Li("seller_type_Trustmark Dealer"),
                html.Li("owner_2 (Second owner)"),
                html.Li("owner_3 (Third owner)"),
                html.Li("owner_4 (Fourth & Above Owner)"),
            ]
        ),

        html.H3("Models available:"),
        html.Ul(
            [
                html.Li("Model A1: RandomForestRegressor"),
                html.Li("Model A2: Polynomial") ,
                html.Li("Model A3: LogisticRegression"),
            ]
        ),

        html.Hr(),
        dcc.Markdown(
            """
**How to use the app**
- Go to the **Prediction** page.
- Enter the car details.
- Click **Predict** to see the result.

**********************************************************************************
Aphisit Jaemyaem — st126130
"""
        ),
        dcc.Link(html.Button("Go to Prediction page"), href="/prediction_model_2"),
    ],
    style={"padding": "20px"},
)
