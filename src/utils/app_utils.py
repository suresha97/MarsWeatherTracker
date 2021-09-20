import os
import joblib
import json

import plotly.express as px
import pandas as pd
import torch

from aws_rds.mars_weather_data_rds_database import MarsWeatherDataRDSDatabase
from forecasting_models.forecasting_model_data import ForecastingDataModel
from forecasting_models.models.rnn_model import RNNModel


_MODEL_SAVE_PATH = "../trained_models"


def make_plotly_graph(parameters):
    if parameters["display_forecast"] is not None and "Display Forecast" in parameters["display_forecast"]:
        parameters["plot"]["y"].append("predictions")
        parameters["column_labels"].append(f"Forecasted {get_quantity_value_from_label(parameters['plot']['y'][0])}")

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


def get_weather_forecasts(weather_data, model_type, quantity):
    features, labels = prepare_data_for_model(weather_data, quantity)
    learned_model = load_latest_trained_model(model_type, quantity)
    #learned_model = OrderedDict({k[1:]: v for k, v in learned_model.items()})
    rnn_forecaster = RNNModel(load_model_architecture_inputs(model_type, quantity))
    rnn_forecaster.load_state_dict(learned_model)
    scaled_predictions = rnn_forecaster.forward(features)
    _, scaler_y = load_scalers(quantity)

    return scaler_y.inverse_transform(scaled_predictions.detach())


def prepare_data_for_model(weather_data, quantity):
    weather_data.terrestrial_date = pd.to_datetime(weather_data.terrestrial_date)
    dataset_parameters = {
        "start_date": weather_data.terrestrial_date.min(),
        "end_date": weather_data.terrestrial_date.max(),
        "quantity_to_forecast": quantity,
        "training_data_prop": 0.0,
        "val_data_prop": 0.0,
        "num_lags": 10,
        "num_timesteps_to_forecast": 1,
        "scaling_type": None,
    }

    forecasting_data_model = ForecastingDataModel(
        weather_data,
        dataset_parameters["start_date"],
        dataset_parameters["end_date"],
        dataset_parameters["quantity_to_forecast"]
    )

    forecasting_data = forecasting_data_model.get_torch_training_data(
        dataset_parameters["training_data_prop"],
        dataset_parameters["val_data_prop"],
        dataset_parameters["num_lags"],
        dataset_parameters["num_timesteps_to_forecast"],
        dataset_parameters["scaling_type"]
    )

    scaled_features, scaled_labels = get_scaled_features_and_labels(forecasting_data, quantity)

    return torch.tensor(scaled_features, dtype=torch.float), torch.tensor(scaled_labels, dtype=torch.float)


def get_scaled_features_and_labels(forecasting_data, quantity):
    features, labels = forecasting_data["test_data"]
    features_reshape = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
    scaler_x, scaler_y = load_scalers(quantity)

    return scaler_x.transform(features_reshape).reshape((features.shape[0], features.shape[1], features.shape[2])), scaler_y.transform(labels)


def load_scalers(quantity):
    return (
        joblib.load(f"{_MODEL_SAVE_PATH}/{quantity}/scalers/normalisation_features.gz"),
        joblib.load(f"{_MODEL_SAVE_PATH}/{quantity}/scalers/normalisation_labels.gz")
    )


def load_latest_trained_model(model_type, quantity):
    saved_models = [
        file_name for file_name in os.listdir(os.path.abspath(f"{_MODEL_SAVE_PATH}/{quantity}")) if model_type in file_name
    ]
    sorted_models = sorted(saved_models)

    return torch.load(f"{_MODEL_SAVE_PATH}/{quantity}/{sorted_models[0]}")


def load_model_architecture_inputs(model_type, quantity):
    saved_models = [
        file_name for file_name in os.listdir(os.path.abspath(f"{_MODEL_SAVE_PATH}/{quantity}"))
        if model_type in file_name and "inputs" in file_name
    ]

    sorted_models = sorted(saved_models)

    with open(f"{_MODEL_SAVE_PATH}/{quantity}/{sorted_models[0]}") as f:
        inputs = json.load(f)

    return inputs


def get_quantity_value_from_label(quantity):
    quantity_to_label_map = {
        "min_temp": "Temperature",
        "max_temp": "Temperature",
        "min_gts_temp": "Temperature",
        "max_gts_temp": "Temperature",
        "pressure": "Pressure"
    }

    return quantity_to_label_map[quantity]