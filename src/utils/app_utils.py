from typing import Dict, Any, Tuple, Union
import os
import joblib
import json
from collections import OrderedDict

import numpy as np
import plotly.express as px
import pandas as pd
import torch
from torch import Tensor
from matplotlib.pyplot import Figure

from aws_rds.mars_weather_data_rds_database import MarsWeatherDataRDSDatabase
from forecasting_models.forecasting_model_data import ForecastingDataModel
from forecasting_models.models.rnn_model import RNNModel
from forecasting_models.models.lstm_model import LSTMModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler


_MODEL_SAVE_PATH = "../trained_models"


def make_plotly_graph(parameters: Dict[str, Any]) -> Figure:
    if parameters["display_forecast"] is not None and "Display Forecast" in parameters["display_forecast"]:
        parameters["plot"]["y"].append("predictions")
        parameters["column_labels"].append(f"Forecasted {get_quantity_value_from_label(parameters['plot']['y'][0])}")

    if parameters["plot_type"] == "line":
        fig = px.line(**parameters["plot"])

    elif parameters["plot_type"] == "scatter":
        if parameters["colormap_quantity"] is not None:
            parameters["plot"]["color"] = parameters["colormap_quantity"]
        fig = px.scatter(**parameters["plot"])

        if parameters["colormap_quantity"] is not None:
            fig.update_layout(
                coloraxis_colorbar={
                    "title": get_quantity_value_from_label(parameters["colormap_quantity"])
                }
            )

    for i, new_name in enumerate(parameters["column_labels"]):
        fig.data[i].name = new_name

    fig.update_layout(
        showlegend=False
    )

    return fig


def get_latest_mars_weather_data() -> pd.DataFrame:
    mars_weather_data_rds_database = MarsWeatherDataRDSDatabase(
        f"{os.environ.get('USER')}",
        f"{os.environ.get('PASSWORD')}",
        f"{os.environ.get('DATABASE_HOST')}",
        int(os.environ.get('DATABASE_PORT')),
        f"{os.environ.get('DATABASE_NAME')}"
    )

    mars_weather_data_df = mars_weather_data_rds_database.query_data_from_table_into_dataframe(
        "SELECT * FROM curiosity_mars_weather_data"
    )

    mars_weather_data_df.sort_values("terrestrial_date", inplace=True)

    return mars_weather_data_df


def get_weather_forecasts(weather_data: pd.DataFrame, model_type: str, quantity: str) -> np.ndarray:
    features, labels = prepare_data_for_model(weather_data, quantity)
    learned_model = load_latest_trained_model(model_type, quantity)
    forecaster = _get_forecaster(model_type, quantity)
    forecaster.load_state_dict(learned_model)
    scaled_predictions = forecaster.forward(features)
    _, scaler_y = load_scalers(quantity)

    return scaler_y.inverse_transform(scaled_predictions.detach())


def prepare_data_for_model(weather_data: pd.DataFrame, quantity: str) -> Tuple[Tensor, Tensor]:
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


def get_scaled_features_and_labels(
        forecasting_data: Dict[str, Tuple[Tensor, Tensor]], quantity: str
) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = forecasting_data["test_data"]
    features_reshape = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
    scaler_x, scaler_y = load_scalers(quantity)

    return scaler_x.transform(features_reshape).reshape((features.shape[0], features.shape[1], features.shape[2])), scaler_y.transform(labels)


def load_scalers(quantity: str) -> Tuple[Union[MinMaxScaler, StandardScaler], Union[MinMaxScaler, StandardScaler]]:
    return (
        joblib.load(f"{_MODEL_SAVE_PATH}/{quantity}/scalers/normalisation_features.gz"),
        joblib.load(f"{_MODEL_SAVE_PATH}/{quantity}/scalers/normalisation_labels.gz")
    )


def load_latest_trained_model(model_type: str, quantity: str) -> OrderedDict:
    saved_models = [
        file_name for file_name in os.listdir(os.path.abspath(f"{_MODEL_SAVE_PATH}/{quantity}")) if model_type in file_name
    ]
    sorted_models = sorted(saved_models)

    return torch.load(f"{_MODEL_SAVE_PATH}/{quantity}/{sorted_models[0]}")


def load_model_architecture_inputs(model_type: str, quantity: str) -> Dict[str, Any]:
    saved_models = [
        file_name for file_name in os.listdir(os.path.abspath(f"{_MODEL_SAVE_PATH}/{quantity}"))
        if model_type in file_name and "inputs" in file_name
    ]

    sorted_models = sorted(saved_models)

    with open(f"{_MODEL_SAVE_PATH}/{quantity}/{sorted_models[0]}") as f:
        inputs = json.load(f)

    return inputs


def get_quantity_value_from_label(quantity: str) -> str:
    quantity_to_label_map = {
        "min_temp": "Minimum Air Temperature (C)",
        "max_temp": "Maximum Air Temperature (C)",
        "min_gts_temp": "Minimum Ground Temperature (C)",
        "max_gts_temp": "Maximum Ground Temperature (C)",
        "pressure": "Atmospheric Pressure (Pa)"
    }

    return quantity_to_label_map[quantity]


def _get_forecaster(model_type: str, quantity: str) -> Union[RNNModel, LSTMModel]:
    model_type_map = {
        "rnn": RNNModel,
        "lstm": LSTMModel
    }

    forecaster = model_type_map[model_type]

    return forecaster(load_model_architecture_inputs(model_type, quantity))
