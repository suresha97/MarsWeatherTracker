from sqlalchemy import create_engine
import pandas as pd

from data_collectors.utils import load_json_from_s3_as_dict
from aws_lambda.save_processed_weather_data_to_rds_table import process_raw_mars_weather_data


class MarsWeatherDataRDSDatabase:
    def __init__(self, user, password, host, port, database_name):
        self.user = user
        self.database_password = password
        self.host = host
        self.port = port
        self.database_name = database_name
        self.database_engine = self.get_database_engine()

    def get_database_engine(self):
        mars_weather_databse = create_engine(
            f"postgresql+psycopg2://"
            f"{self.user}:"
            f"{self.database_password}"
            f"@{self.host}:"
            f"{self.port}/"
            f"{self.database_name}"
        )

        return mars_weather_databse

    def create_new_table_from_dataframe(self, dataframe, table_name):
        dataframe.to_sql(table_name, self.database_engine, if_exists="replace", index=False)

    def query_data_from_table_into_dataframe(self, sql_query):
        database_connection = self.database_engine.connect()
        query_output = pd.read_sql(sql_query, database_connection)

        return query_output

    def add_entries_to_table(self, dataframe, table_name):
        dataframe.to_sql(table_name, self.database_engine, if_exists="append", index=False)


if __name__ == "__main__":
    s3_bucket = "mars-weather-data"
    s3_key = "raw_mars_weather_data.json"

    raw_mars_weather_data = load_json_from_s3_as_dict(s3_bucket, s3_key)
    mars_weather_data = process_raw_mars_weather_data(raw_mars_weather_data)
    print(mars_weather_data.head())
    print(mars_weather_data.shape)
    print(mars_weather_data.columns)

    test_rds = MarsWeatherDataRDSDatabase("postgres", "Unn11997*", "mars-weather-data.c0wmlpzuprn2.eu-west-1.rds.amazonaws.com", 5432, "curiosity_mars_weather_data" )

    test_rds_engine = test_rds.get_database_engine()
    mars_weather_data.to_sql("mars_weather_data", test_rds_engine, if_exists="append", index=False)
    print("Querying data from RDS...")
    query = "SELECT * FROM mars_weather_data"
    query_data = test_rds.query_data_from_table_into_dataframe(query)
    print(type(query_data))
    print(query_data.head())
    print(query_data.shape)
    print(query_data.columns)
