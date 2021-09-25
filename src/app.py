from datetime import date

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from matplotlib.pyplot import Figure

from assets.app_style import SIDEBAR_STYLE, CONTENT_STYLE
from utils.app_utils import get_latest_mars_weather_data, make_plotly_graph, get_weather_forecasts, \
    get_quantity_value_from_label


dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

sidebar = html.Div(
    [
        html.Br(),
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
            style={'width': "95%", "margin-left": "0.5rem", "font-size": 16},
            placeholder="Select Quantity"
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.Dropdown(
            id="select_plot_type",
            options=[
                {"label": "Line Plot", "value": "line"},
                {"label": "Scatter Plot", "value": "scatter"},
            ],
            multi=False,
            style={'width': "95%", "margin-left": "0.5rem", "font-size": 16},
            placeholder="Select Visualisation Method"
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.Dropdown(
            id="select_colormap_quantity",
            options=[
                {"label": "Min Air Temperature", "value": "min_temp"},
                {"label": "Max Air Temperature", "value": "max_temp"},
                {"label": "Min Ground Temperature", "value": "min_gts_temp"},
                {"label": "Max Ground Temperature", "value": "max_gts_temp"},
                {"label": "Pressure", "value": "pressure"}
            ],
            multi=False,
            style={'width': "95%", "margin-left": "0.5rem", "font-size": 16},
            placeholder="Select Colormap Quantity",
            value=None
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
        dbc.Row(dcc.Dropdown(
            id="select_model",
            options=[
                {"label": "RNN", "value": "rnn"},
                {"label": "LSTM", "value": "lstm"},
            ],
            multi=False,
            style={'width': "95%", "margin-left": "0.5rem", "font-size": 16},
            placeholder="Select Forecasting Model"
        )),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(dcc.Checklist(
            id="display_forecast",
            options=[
                {"label": "Display Forecast", "value": "Display Forecast"}
            ],
            style={'width': "90%", "margin-left": "1rem", "font-size": 16},
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
     Input(component_id="select_plot_type", component_property="value"),
     Input(component_id="select_colormap_quantity", component_property="value"),
     Input(component_id="start_date", component_property="date"),
     Input(component_id="end_date", component_property="date"),
     Input(component_id="select_model", component_property="value"),
     Input(component_id="display_forecast", component_property="value")]
)
def update_graph(
        selected_quantity: str,
        plot_type: str,
        colormap_quantity: str,
        start_date: str,
        end_date: str,
        model_type: str,
        display_forecast: str,
) -> Figure:
    mars_weather_data_df = get_latest_mars_weather_data()

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

    fig = make_plotly_graph(
        {
            "plot_type": plot_type,
            "plot": {
                "data_frame": mars_weather_data_filtered,
                "x": "terrestrial_date",
                "y": [selected_quantity],
                "labels": {
                    "terrestrial_date": "Date",
                    "value": quantity_value_to_label_map
                },
                "template": "plotly_dark"
            },
            "column_labels": [f"Observed {quantity_value_to_label_map}"],
            "display_forecast": display_forecast,
            "colormap_quantity": colormap_quantity
        }
    )

    return fig


if __name__ == "__main__":
    dash_app.run_server(debug=False, port=3004)
