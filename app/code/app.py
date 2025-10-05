from dash import Dash, html, page_container
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server
app.title = "Car Price Prediction"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Prediction", href="/prediction_model_2")),
    ],
    brand="", brand_href="/", color="success", dark=True,
)

app.layout = html.Div([navbar, page_container])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8000, debug=False)
