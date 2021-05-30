import requests

from data_collectors.utils import save_json_to_s3

API_BASE_URL = "https://mars.nasa.gov/rss/api"


class CuriosityRoverWeatherDataCollector:
    def __init__(self, raw_mars_weather_data_s3_save_bucket, raw_mars_weather_data_s3_save_key):
        self._raw_mars_weather_data_s3_save_bucket = raw_mars_weather_data_s3_save_bucket
        self._raw_mars_weather_data_s3_save_key = raw_mars_weather_data_s3_save_key

    def save_raw_mars_weather_data_to_s3(self):
        raw_mars_weather_data = self._retreive_raw_mars_weather_data_from_api()
        save_json_to_s3(
            raw_mars_weather_data,
            self._raw_mars_weather_data_s3_save_bucket,
            self._raw_mars_weather_data_s3_save_key
        )

    def _retreive_raw_mars_weather_data_from_api(self):
        parameters = {
            "feed": "weather",
            "category": "msl",
            "feedtype": "json"
        }
        raw_mars_weather_data = requests.get(API_BASE_URL, params=parameters)

        return raw_mars_weather_data.json()
