from typing import Dict, Any
import requests

from utils.s3_utils import save_json_to_s3

API_BASE_URL = "https://mars.nasa.gov/rss/api"


class CuriosityRoverWeatherDataCollector:
    def __init__(self, raw_mars_weather_data_s3_save_bucket: str, raw_mars_weather_data_s3_save_key: str) -> None:
        self._raw_mars_weather_data_s3_save_bucket = raw_mars_weather_data_s3_save_bucket
        self._raw_mars_weather_data_s3_save_key = raw_mars_weather_data_s3_save_key

    def save_raw_mars_weather_data_to_s3(self) -> None:
        raw_mars_weather_data = self._retreive_raw_mars_weather_data_from_api()
        save_json_to_s3(
            raw_mars_weather_data,
            self._raw_mars_weather_data_s3_save_bucket,
            self._raw_mars_weather_data_s3_save_key
        )

    def _retreive_raw_mars_weather_data_from_api(self) -> Dict[str, Any]:
        parameters = {
            "feed": "weather",
            "category": "msl",
            "feedtype": "json"
        }

        raw_mars_weather_data = requests.get(API_BASE_URL, params=parameters)

        return raw_mars_weather_data.json()
