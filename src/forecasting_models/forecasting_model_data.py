import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ForecastingDataModel:
    def __init__(self, weather_data, start_date, end_date, quantity_to_forecast):
        self._weather_data = weather_data
        self._start_date = start_date
        self._end_date = end_date
        self._quantity_to_forecast = quantity_to_forecast

    def get_torch_training_data(self, train_data_proportion, val_data_prop, num_lags, num_timsteps_to_forecast, scaling_type):
        relevant_weather_data = self._get_relevant_weather_data()

        relevant_weather_data_with_temporal_features = add_temporal_features_to_dataset(relevant_weather_data)

        train_data, val_data, test_data = get_train_val_test_split(
            relevant_weather_data_with_temporal_features[["timestamp_sin", "timestamp_cos", self._quantity_to_forecast]],
            train_data_proportion,
            val_data_prop
        )

        features_and_labels = self._get_features_and_labels(
            (train_data, val_data, test_data),
            num_lags,
            num_timsteps_to_forecast
        )

        scaled_features_and_labels = get_scaled_features_and_labels(features_and_labels, scaling_type)

        return {
            "train_data": (
                create_torch_tensors(scaled_features_and_labels["train"]["features"]),
                create_torch_tensors(scaled_features_and_labels["train"]["labels"])
            ),
            "validation_data": (
                create_torch_tensors(scaled_features_and_labels["validation"]["features"]),
                create_torch_tensors(scaled_features_and_labels["validation"]["labels"])
            ),
            "test_data": (
                create_torch_tensors(scaled_features_and_labels["test"]["features"]),
                create_torch_tensors(scaled_features_and_labels["test"]["labels"])
            )
        }

    def _get_relevant_weather_data(self):
        self._weather_data["terrestrial_date"] = pd.to_datetime(self._weather_data["terrestrial_date"])
        relevant_weather_data = self._weather_data[
            (self._weather_data.terrestrial_date > self._start_date) &
            (self._weather_data.terrestrial_date < self._end_date)
        ]

        return relevant_weather_data[["terrestrial_date", self._quantity_to_forecast]]

    def _get_features_and_labels(self, data_splits, num_lags, num_timesteps_to_forecast):
        train_data, val_data, test_data = data_splits
        train_x, train_y = create_training_sequences_and_labels(
            train_data, self._quantity_to_forecast, num_lags=num_lags, num_timesteps_to_forecast=num_timesteps_to_forecast
        )

        val_x, val_y = create_training_sequences_and_labels(
            val_data, self._quantity_to_forecast, num_lags=num_lags, num_timesteps_to_forecast=num_timesteps_to_forecast
        )

        test_x, test_y = create_training_sequences_and_labels(
            test_data, self._quantity_to_forecast, num_lags=num_lags, num_timesteps_to_forecast=num_timesteps_to_forecast
        )

        return {
            "train": {
                "features": train_x,
                "labels": train_y
            },
            "validation": {
                "features": val_x,
                "labels": val_y,
            },
            "test": {
                "features": test_x,
                "labels": test_y
            }
        }


def add_temporal_features_to_dataset(weather_data):
    weather_data["timestamp"] = weather_data.terrestrial_date.apply(lambda x: x.timestamp())

    num_seconds_in_an_earth_year = 60 * 60 * 24 * 365.25

    def sine_transformation(x):
        return np.sin(
            (2 * np.pi * x) / num_seconds_in_an_earth_year
        )

    def cos_transformation(x):
        return np.cos(
            (2 * np.pi * x) / num_seconds_in_an_earth_year
        )

    weather_data["timestamp_sin"] = weather_data.timestamp.apply(sine_transformation)
    weather_data["timestamp_cos"] = weather_data.timestamp.apply(cos_transformation)

    return weather_data


def get_train_val_test_split(data, train_prop, val_prop):
    train_data, test_data = create_train_test_split(data, train_prop=train_prop)
    val_data, test_data = create_train_test_split(test_data, train_prop=val_prop)

    return train_data, val_data, test_data


def create_train_test_split(data, train_prop):
    train_data_df = data.iloc[:int(len(data) * train_prop)]
    test_data_df = data.iloc[len(train_data_df):]

    return train_data_df, test_data_df


def create_training_sequences_and_labels(data, quantity_name, num_lags, num_timesteps_to_forecast):
    train_x = []
    train_y = []

    df_reset_index = data.reset_index(drop=True)
    quantity_series = df_reset_index[quantity_name].values

    for i in range(len(quantity_series) - num_lags - num_timesteps_to_forecast):
        seq_x = df_reset_index.iloc[i:(i + num_lags)].values
        seq_y = quantity_series[(i + num_lags):(i + num_lags + num_timesteps_to_forecast)]

        train_x.append(seq_x)
        train_y.append(seq_y)

    return np.array(train_x), np.array(train_y)


def get_scaled_features_and_labels(features_and_labels, scaling_type):
    train_x_scaled, val_x_scaled, test_x_scaled = scaled_features(
        features_and_labels["train"]["features"],
        features_and_labels["validation"]["features"],
        features_and_labels["test"]["features"],
        scaling_type=scaling_type
    )

    train_y_scaled, val_y_scaled, test_y_scaled = scaled_labels(
        features_and_labels["train"]["labels"],
        features_and_labels["validation"]["labels"],
        features_and_labels["test"]["labels"],
        scaling_type=scaling_type
    )

    return {
            "train": {
                "features": train_x_scaled,
                "labels": train_y_scaled
            },
            "validation": {
                "features": val_x_scaled,
                "labels": val_y_scaled,
            },
            "test": {
                "features": test_x_scaled,
                "labels": test_y_scaled
            }
        }


def scaled_features(features_train, features_val, features_test, scaling_type):
    scaler = scaling_type_to_scaler_map()[scaling_type]

    seq_len, num_features = features_train.shape[1], features_train.shape[2]

    train_x_reshape = features_train.reshape((features_train.shape[0] * seq_len, num_features))
    val_x_reshape = features_val.reshape((features_val.shape[0] * seq_len, num_features))
    test_x_reshape = features_test.reshape((features_test.shape[0] * seq_len, num_features))

    scaler.fit(train_x_reshape)

    train_x_scaled = scaler.transform(train_x_reshape).reshape((features_train.shape[0], seq_len, num_features))
    val_x_scaled = scaler.transform(val_x_reshape).reshape((features_val.shape[0], seq_len, num_features))
    test_x_scaled = scaler.transform(test_x_reshape).reshape((features_test.shape[0], seq_len, num_features))

    return train_x_scaled, val_x_scaled, test_x_scaled


def scaled_labels(labels_train, labels_val, labels_test, scaling_type):
    scaler_y = scaling_type_to_scaler_map()[scaling_type]
    scaler_y.fit(labels_train)

    return scaler_y.transform(labels_train), scaler_y.transform(labels_val), scaler_y.transform(labels_test)


def scaling_type_to_scaler_map():
    return {
        "normalisation": MinMaxScaler(),
        "standardisation": StandardScaler()
    }


def create_torch_tensors(data):
    return torch.tensor(data, dtype=torch.float)
