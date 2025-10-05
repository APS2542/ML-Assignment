from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server

app.layout = dbc.Container(
    [
        html.H2("A3 App (clean start)"),
        html.P("Pipeline: MLflow re-log -> Tests -> Build/Push -> Deploy"),
        html.P("This app intentionally minimal and independent from model."),
        html.P("Deployed container listens on port 8000 (published 8080)."),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8000, debug=False)
