from datetime import date
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.app_utils import make_plotly_graph, get_latest_mars_weather_data

mars_weather_data_df = get_latest_mars_weather_data()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Mars Weather Tracker", style={'text-align': 'center'}),

    dcc.Dropdown(id="select_quantity",
                     options=[
                         {"label": "Air Temperature", "value": "air_temp"},
                         {"label": "Ground Temperature", "value": "ground_temp"},
                         {"label": "Pressure", "value": "pressure"}
                     ],
                     multi=False,
                     value="air_temp",
                     style={'width': "40%"}
                     ),

    html.Br(),

    dcc.DatePickerRange(
        id='date_range',
        min_date_allowed=date(2013, 1, 1),
        max_date_allowed=date(2021, 5, 1),
        initial_visible_month=date(2017, 8, 5),
        end_date=date.today()
    ),

    html.Br(),

    dcc.Graph(id="mars_weather_data_over_time_graph", figure={})

])

@app.callback(
    Output(component_id="mars_weather_data_over_time_graph", component_property="figure"),
    [Input(component_id="select_quantity", component_property="value"),
     Input(component_id="date_range", component_property="start_date"),
     Input(component_id="date_range", component_property="end_date")]
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
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["min_temp", "max_temp"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "temperature"
                    }
                },
                "column_labels": ["Min Temperature", "Max Temperature"]
            }
        )

    if selected_quantity == "ground_temp":
        fig = make_plotly_graph(
            {
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["min_gts_temp", "max_gts_temp"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "temperature"
                    }
                },
                "column_labels": ["Min Ground Temperature", "Max Ground Temperature"]
            }
        )

    if selected_quantity == "pressure":
        fig = make_plotly_graph(
            {
                "plot": {
                    "data_frame": mars_weather_data_filtered,
                    "x": "terrestrial_date",
                    "y": ["pressure"],
                    "labels": {
                        "terrestrial_date": "Date",
                        "value": "pressure"
                    }
                },
                "column_labels": ["Pressure"]
            }
        )

    return fig


### BUG : Y-AXIS scale messed up when plotting multiplt lines
if __name__ == "__main__":
    app.run_server(debug=True)
