import time

import torch.nn as nn
import torch.optim as optim

from forecasting_models.model_trainers.torch_trainer import TorchTrainer
from forecasting_models.models.rnn_model import RNNModel


class RNNTrainer(TorchTrainer):
    def __init__(self, training_parameters, forecasting_data, quantity_to_forecast):
        super().__init__(training_parameters, forecasting_data, quantity_to_forecast)
        self._quantity_to_forecast = quantity_to_forecast
        self._output_size = 1
        self._input_size = self._forecasting_data["train_data"][0].size(-1)
        self._hidden_size = self._training_parameters["hidden_layer_size"]
        self._num_layers = self._training_parameters["num_hidden_layers"]

        self._model_architecture_inputs = {
            "input_size": self._input_size,
            "output_size": self._output_size,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers
        }

        self._model = RNNModel(self._model_architecture_inputs)

    def _define_loss_criterion(self):
        return nn.MSELoss()

    def _initialise_optimiser(self):
        return optim.AdamW(
            self.parameters(),
            lr=self._training_parameters["learning_rate"],
            weight_decay=self._training_parameters["weight_decay"]
        )

    @property
    def _model_save_name(self):
        return "rnn_model"
