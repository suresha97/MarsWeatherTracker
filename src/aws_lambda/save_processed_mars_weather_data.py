import pandas as pd
import numpy as np

from utils import load_json_from_s3_as_dict
from aws_rds.mars_weather_data_rds_database import MarsWeatherDataRDSDatabase


def process_raw_mars_weather_data(raw_mars_weather_data):
    raw_mars_weather_data_by_sol = raw_mars_weather_data["soles"]
    mars_weather_df_columns = list(raw_mars_weather_data_by_sol[0].keys())
    mars_weather_df = pd.DataFrame(raw_mars_weather_data_by_sol, columns=mars_weather_df_columns)

    mars_weather_df.replace("--", np.nan, inplace=True)
    mars_weather_df.dropna(inplace=True, subset=["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp"])
    mars_weather_df[["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp"]] = mars_weather_df[
        ["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp"]
    ].apply(pd.to_numeric)

    return mars_weather_df


def lambda_handler(event, context):
    raw_mars_weather_data_s3_bucket = "mars-weather-data"
    raw_mars_weather_data_s3_key = "raw_mars_weather_data.json"

    raw_mars_weather_data = load_json_from_s3_as_dict(raw_mars_weather_data_s3_bucket, raw_mars_weather_data_s3_key)
    mars_weather_data = process_raw_mars_weather_data(raw_mars_weather_data)

    mars_weather_data_rds_database = MarsWeatherDataRDSDatabase(
        "postgres",
        "Unn11997*",
        "mars-weather-data.c0wmlpzuprn2.eu-west-1.rds.amazonaws.com",
        5432,
        "curiosity_mars_weather_data"
    )
    mars_weather_data_rds_database.create_new_table_from_dataframe(mars_weather_data, "curiosity_mars_weather_data")

    return


if __name__ == "__main__":
    lambda_handler({}, None)
