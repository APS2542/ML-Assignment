from dash import Dash, html, page_container
import dash_bootstrap_components as dbc
from model.model import *

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CERULEAN])
app.title = "Car Price Prediction (New!!)"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        # dbc.NavItem(dbc.NavLink("Model 1", href="/prediction_model")),
        dbc.NavItem(dbc.NavLink("Prediction", href="/prediction_model_2")),
    ],
    brand="",
    brand_href="/",
    color="success",
    dark=True,
)

app.layout = html.Div([navbar, page_container])
#Run app
if __name__ == "__main__":
    app.run(debug=True)


