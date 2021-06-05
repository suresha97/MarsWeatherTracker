from data_collectors.mars_weather.curiosity_rover_weather_data_collector import CuriosityRoverWeatherDataCollector


def lambda_handler(event, context):
    raw_mars_weather_data_s3_save_bucket = "mars-weather-data"
    raw_mars_weather_data_s3_save_key = "raw_mars_weather_data.json"
    curiosity_rover_weather_data_collector = CuriosityRoverWeatherDataCollector(
        raw_mars_weather_data_s3_save_bucket,
        raw_mars_weather_data_s3_save_key
    )

    curiosity_rover_weather_data_collector.save_raw_mars_weather_data_to_s3()

    return


if __name__ == "__main__":
    lambda_handler({}, None)
