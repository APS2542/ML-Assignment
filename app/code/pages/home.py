import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    [
        html.H2("Welcome to Car Price Prediction App 🚗"),
        html.P("This app predicts the estimated selling price of a car "),

        html.H3("Features used in the model:"),
        html.Ul(
            [
                html.Li("year (manufacturing year)"),
                html.Li("km_driven (kilometers driven)"),
                html.Li("mileage (fuel efficiency, km/l)"),
                html.Li("max_power (engine power in bhp)"),
                html.Li("fuel_Petrol"),
                html.Li("transmission_Manual"),
                html.Li("seller_type_Individual "),
                html.Li("seller_type_Trustmark Dealer "),
                html.Li("owner_2 (Second owner)"),
                html.Li("owner_3 (Third owner)"),
                html.Li("owner_4 (Fourth & Above Owner)"),
            ]
        ),

        html.H3("Model used:"),
        html.P("RandomForestRegressor"),

        html.Hr(),
        dcc.Markdown(
            """
            **How to use the app:**
            - Go to the *Prediction Model* page.
            - Enter the car details.
            - Click **Predict Price** to see the estimated selling price.
             **********************************************************************************
            Aphisit Jaemyaem st126130
            """
        ),
    ],
    style={"padding": "20px"},
)
