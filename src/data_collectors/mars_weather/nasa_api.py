import requests
import pandas as pd

API_BASE_URL = "https://mars.nasa.gov/rss/api"

class NasMarsWeatherServiceApiCollector:
    def __init__(self):
        pass

    def _convert_weather_data_to_dataframe(self, json):
        raw_mars_weather_df = pd.DataFrame()
        return

    def _retreive_data_from_api(self):
        parameters = {
            "feed": "weather",
            "category": "msl",
            "feedtype": "json"
        }
        response = requests.get(API_BASE_URL, params=parameters)
        raw_mars_weather_data = response.json()["soles"]
        print(type(raw_mars_weather_data[0]))
        return raw_mars_weather_data

if __name__ == "__main__":
    mars_weather = NasMarsWeatherServiceApiCollector()
    mars_weather._retreive_data_from_api()