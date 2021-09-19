import plotly.express as px

from src.aws_rds.mars_weather_data_rds_database import MarsWeatherDataRDSDatabase


def make_plotly_graph(parameters):
    if parameters["plot_type"] == "line":
        fig = px.line(**parameters["plot"])
        for i, new_name in enumerate(parameters["column_labels"]):
            fig.data[i].name = new_name
    elif parameters["plot_type"] == "scatter":
        fig = px.scatter(**parameters["plot"])

    return fig


def get_latest_mars_weather_data():
    mars_weather_data_rds_database = MarsWeatherDataRDSDatabase(
        "postgres",
        "Unn11997*",
        "mars-weather-data.c0wmlpzuprn2.eu-west-1.rds.amazonaws.com",
        5432,
        "curiosity_mars_weather_data"
    )

    mars_weather_data_df = mars_weather_data_rds_database.query_data_from_table_into_dataframe(
        "SELECT * FROM curiosity_mars_weather_data"
    )

    mars_weather_data_df.sort_values("terrestrial_date", inplace=True)

    return mars_weather_data_df