from sqlalchemy import create_engine
import pandas as pd


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
