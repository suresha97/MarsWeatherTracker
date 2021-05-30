import pandas as pd

def process_raw_mars_weather_data(raw_mars_weather_data):
    raw_mars_weather_data_by_sol = raw_mars_weather_data["soles"]
    mars_weather_df_columns = list(raw_mars_weather_data_by_sol[0].keys())
    mars_weather_df = pd.DataFrame(raw_mars_weather_data_by_sol, columns=mars_weather_df_columns)

    return mars_weather_df
