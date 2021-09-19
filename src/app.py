from datetime import date

import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app_utils import make_plotly_graph

#mars_weather_data_df = get_latest_mars_weather_data()
#mars_weather_data_df.to_csv("local_datasets/mars_weather_data_cleaned.csv")
mars_weather_data_df = pd.read_csv("local_datasets/mars_weather_data_cleaned.csv")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

app.layout = html.Div(
    [
        html.H1("Mars Weather Tracker", style={'text-align': 'center'}),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(
                    id="select_quantity",
                    options=[
                         {"label": "Air Temperature", "value": "air_temp"},
                         {"label": "Ground Temperature", "value": "ground_temp"},
                         {"label": "Pressure", "value": "pressure"}
                    ],
                    multi=False,
                    style={'width': "60%", "offset": 1},
                    placeholder="Quantity"
                )),
                dbc.Col(dcc.DatePickerSingle(
                    id='start_date',
                    min_date_allowed=date(2013, 1, 1),
                    max_date_allowed=date.today(),
                    placeholder="Start Date"
                )),
                dbc.Col(dcc.DatePickerSingle(
                    id='end_date',
                    min_date_allowed=date(2013, 1, 1),
                    max_date_allowed=date.today(),
                    placeholder="End Date"
                ))
            ],
            no_gutters=True
        ),

        html.Br(),
        html.Br(),

        dcc.Graph(id="mars_weather_data_over_time_graph", figure={})
    ]
)

@app.callback(
    Output(component_id="mars_weather_data_over_time_graph", component_property="figure"),
    [Input(component_id="select_quantity", component_property="value"),
     Input(component_id="start_date", component_property="date"),
     Input(component_id="end_date", component_property="date")]
)
def update_graph(selected_quantity, start_date, end_date):
    mars_weather_data_filtered = mars_weather_data_df.copy()
    mars_weather_data_filtered = mars_weather_data_df[
        (mars_weather_data_filtered.terrestrial_date >= start_date) &
        (mars_weather_data_filtered.terrestrial_date <= end_date)
    ]
    print(mars_weather_data_filtered.head())
    if selected_quantity == "air_temp":
        fig = make_plotly_graph(
            {
                "plot_type": "scatter",
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["min_temp", "max_temp"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "Temperature"
                    },
                    "color": "pressure"
                },
                "column_labels": ["Min Temperature", "Max Temperature"]
            }
        )

    if selected_quantity == "ground_temp":
        fig = make_plotly_graph(
            {
                "plot_type": "line",
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["min_gts_temp", "max_gts_temp"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "Temperature"
                    },
                },
                "column_labels": ["Min Ground Temperature", "Max Ground Temperature"]
            }
        )

    if selected_quantity == "pressure":
        fig = make_plotly_graph(
            {
                "plot_type": "line",
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["pressure"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "Pressure"
                    }
                },
                "column_labels": ["Pressure"]
            }
        )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
