from dash import Dash, html, page_container
import dash_bootstrap_components as dbc


app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CERULEAN])
app.title = "Car Price Prediction"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Prediction Model", href="/prediction_model")),
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


