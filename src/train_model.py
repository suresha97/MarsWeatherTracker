import pandas as pd

from forecasting_models.forecasting_model_data import ForecastingDataModel
from forecasting_models.rnn_forecaster import RNNForecaster
from forecasting_models.lstm_forecaster import LSTMForecaster


dataset_parameters = {
    "start_date": "2013-06-01",
    "end_date": "2018-06-01",
    "quantity_to_forecast": "max_temp",
    "training_data_prop": 0.7,
    "val_data_prop": 0.5,
    "num_lags": 10,
    "num_timesteps_to_forecast": 1,
    "scaling_type": "normalisation",
}

model_training_parameters = {
        "hidden_layer_size": 64,
        "num_hidden_layers": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "num_epochs": 50,
        "batch_size": 32,
        "early_stopping_num_epochs": 30
    }


if __name__ == "__main__":
    start_date = "2013-06-01"
    end_date = "2018-06-01"
    quantity_to_forecast = "max_temp"

    local_weather_dataset = pd.read_csv("../local_datasets/mars_weather_data_cleaned.csv")

    forecasting_data_model = ForecastingDataModel(
        local_weather_dataset,
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

    rnn_forecaster = LSTMForecaster(model_training_parameters, forecasting_data)
    rnn_forecaster.train_model()
    rnn_forecaster.evaluate_model()
    rnn_forecaster.save_model()


