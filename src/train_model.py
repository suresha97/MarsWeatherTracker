import pandas as pd

from forecasting_models.forecasting_model_data import ForecastingDataModel
from forecasting_models.model_trainers.rnn_trainer import RNNTrainer
from forecasting_models.model_trainers.lstm_trainer import LSTMTrainer


dataset_parameters = {
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

model_types = {
    "RNN": RNNTrainer,
    "LSTM": LSTMTrainer
}


if __name__ == "__main__":
    start_date = "2013-06-01"
    end_date = "2018-06-01"
    quantity_to_forecast = "pressure"
    model_type = "LSTM"

    local_weather_dataset = pd.read_csv("../local_datasets/mars_weather_data_cleaned.csv")

    forecasting_data_model = ForecastingDataModel(
        local_weather_dataset,
        start_date,
        end_date,
        quantity_to_forecast
    )

    forecasting_data = forecasting_data_model.get_torch_training_data(
        dataset_parameters["training_data_prop"],
        dataset_parameters["val_data_prop"],
        dataset_parameters["num_lags"],
        dataset_parameters["num_timesteps_to_forecast"],
        dataset_parameters["scaling_type"]
    )

    model = model_types[model_type]
    forecaster = model(model_training_parameters, forecasting_data, quantity_to_forecast)
    forecaster.train_model()
    forecaster.evaluate_model()
    forecaster.save_model()
    forecaster.save_model_architecture()
