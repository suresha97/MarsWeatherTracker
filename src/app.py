from datetime import date

import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from utils.app_utils import make_plotly_graph, get_weather_forecasts, get_quantity_value_from_label

# mars_weather_data_df = get_latest_mars_weather_data()
# mars_weather_data_df.to_csv("local_datasets/mars_weather_data_cleaned.csv")
mars_weather_data_df = pd.read_csv("../local_datasets/mars_weather_data_cleaned.csv")
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
                        {"label": "Min Air Temperature", "value": "min_temp"},
                        {"label": "Max Air Temperature", "value": "max_temp"},
                        {"label": "Min Ground Temperature", "value": "min_gts_temp"},
                        {"label": "Max Ground Temperature", "value": "max_gts_temp"},
                        {"label": "Pressure", "value": "pressure"}
                    ],
                    multi=False,
                    style={'width': "70%", "offset": 1},
                    placeholder="Select Quantity"
                )),
                dbc.Col(dcc.Dropdown(
                    id="color_map",
                    options=[
                        {"label": "Min Air Temperature", "value": "min_temp"},
                        {"label": "Max Air Temperature", "value": "max_temp"},
                        {"label": "Min Ground Temperature", "value": "min_gts_temp"},
                        {"label": "Max Ground Temperature", "value": "max_gts_temp"},
                        {"label": "Pressure", "value": "pressure"}
                    ],
                    multi=False,
                    style={'width': "70%", "offset": 1},
                    placeholder="Color Map"
                )),
                dcc.Checklist(
                    id="display_forecast",
                    options=[
                        {"label": "Display Forecast", "value": "Display Forecast"}
                    ],
                    style={'width': "20%", "offset": 1},
                    labelStyle={'display': 'inline-block'}
                ),
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
     Input(component_id="color_map", component_property="value"),
     Input(component_id="display_forecast", component_property="value"),
     Input(component_id="start_date", component_property="date"),
     Input(component_id="end_date", component_property="date")]
)
def update_graph(selected_quantity, color_map, display_forecast, start_date, end_date):
    print(selected_quantity)
    print(display_forecast)
    print(color_map)
    mars_weather_data_filtered = mars_weather_data_df.copy()
    mars_weather_data_filtered = mars_weather_data_df[
        (mars_weather_data_filtered.terrestrial_date >= start_date) &
        (mars_weather_data_filtered.terrestrial_date <= end_date)
        ]

    if display_forecast is not None and "Display Forecast" in display_forecast:
        model_predictions = get_weather_forecasts(mars_weather_data_filtered, "rnn", selected_quantity)
        mars_weather_data_filtered = mars_weather_data_filtered.iloc[len(mars_weather_data_filtered) - len(model_predictions):]
        mars_weather_data_filtered["predictions"] = model_predictions

    quantity_value_to_label_map = get_quantity_value_from_label(selected_quantity)

    print(mars_weather_data_filtered.columns)

    # Note: Color map can only be used with scatter plot
    fig = make_plotly_graph(
        {
            "plot_type": "line",
            "plot": {
                "data_frame": mars_weather_data_filtered,
                "x": "terrestrial_date",
                "y": [selected_quantity],
                "labels": {
                    "terrestrial_date": "Date",
                    "value": quantity_value_to_label_map
                },
            },
            "column_labels": [f"Observed {quantity_value_to_label_map}"],
            "display_forecast": display_forecast,
            "display_color_map": color_map
        }
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
