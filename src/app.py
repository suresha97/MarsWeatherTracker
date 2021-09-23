from datetime import date

import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from utils.app_utils import make_plotly_graph, get_weather_forecasts, get_quantity_value_from_label

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# the style arguments for the sidebar. We use position:fixed and a fixed width
# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}


# mars_weather_data_df = get_latest_mars_weather_data()
# mars_weather_data_df.to_csv("local_datasets/mars_weather_data_cleaned.csv")
mars_weather_data_df = pd.read_csv("../local_datasets/mars_weather_data_cleaned.csv")

sidebar = html.Div(
    [
        html.Br(),
        html.H2("Controls", className="display-6"),
        html.Hr(),
        dbc.Row(dcc.Dropdown(
            id="select_quantity",
            options=[
                {"label": "Min Air Temperature", "value": "min_temp"},
                {"label": "Max Air Temperature", "value": "max_temp"},
                {"label": "Min Ground Temperature", "value": "min_gts_temp"},
                {"label": "Max Ground Temperature", "value": "max_gts_temp"},
                {"label": "Pressure", "value": "pressure"}
            ],
            multi=False,
            style={'width': "95%", "margin-left": "0.5rem"},
            placeholder="Select Quantity"
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.Dropdown(
            id="select_model",
            options=[
                {"label": "RNN", "value": "rnn"},
                {"label": "LSTM", "value": "lstm"},
            ],
            multi=False,
            style={'width': "95%", "margin-left": "0.5rem"},
            placeholder="Select Forecasting Model"
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.DatePickerSingle(
            id='start_date',
            min_date_allowed=date(2013, 1, 1),
            max_date_allowed=date.today(),
            placeholder="Start Date",
            style={'width': "90%", "margin-left": "1rem"}
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.DatePickerSingle(
            id='end_date',
            min_date_allowed=date(2013, 1, 1),
            max_date_allowed=date.today(),
            placeholder="End Date",
            style={'width': "90%", "margin-left": "1rem"}
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.Checklist(
            id="display_forecast",
            options=[
                {"label": "Display Forecast", "value": "Display Forecast"}
            ],
            style={'width': "90%", "margin-left": "1rem"},
            labelStyle={'display': 'inline-block'}
        )),
    ],
    style=SIDEBAR_STYLE
)

content = html.Div(
    dcc.Graph(id="mars_weather_data_over_time_graph", figure={}, style=CONTENT_STYLE)
)

dash_app.layout = html.Div(
    [
        html.Br(),
        html.H1("Mars Weather Tracker", style={'text-align': 'center'}),
        sidebar,
        html.Br(),
        content,
        dcc.Input(style={"margin-left": "15px"})
    ]
)


@dash_app.callback(
    Output(component_id="mars_weather_data_over_time_graph", component_property="figure"),
    [Input(component_id="select_quantity", component_property="value"),
     Input(component_id="select_model", component_property="value"),
     Input(component_id="display_forecast", component_property="value"),
     Input(component_id="start_date", component_property="date"),
     Input(component_id="end_date", component_property="date")]
)
def update_graph(selected_quantity, model_type, display_forecast, start_date, end_date):
    print(selected_quantity)
    print(display_forecast)

    mars_weather_data_filtered = mars_weather_data_df.copy()
    mars_weather_data_filtered = mars_weather_data_df[
        (mars_weather_data_filtered.terrestrial_date >= start_date) &
        (mars_weather_data_filtered.terrestrial_date <= end_date)
        ]

    if display_forecast is not None and "Display Forecast" in display_forecast:
        model_predictions = get_weather_forecasts(mars_weather_data_filtered, model_type, selected_quantity)
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
            "display_forecast": display_forecast
        }
    )

    return fig


if __name__ == "__main__":
    dash_app.run_server(debug=False, port=3004)
