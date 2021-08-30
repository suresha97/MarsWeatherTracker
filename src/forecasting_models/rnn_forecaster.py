import time

import torch
import torch.nn as nn
import torch.optim as optim

from forecasting_models.torch_forecaster import TorchForecaster


class RNNForecaster(TorchForecaster):
    def __init__(self, training_parameters, forecasting_data):
        super().__init__(training_parameters, forecasting_data)
        self._output_size = 1
        self._input_size = self._forecasting_data["train_data"][0].size(-1)
        self._hidden_size = self._training_parameters["hidden_layer_size"]
        self._num_layers = self._training_parameters["num_hidden_layers"]
        self._rnn = nn.RNN(self._input_size, self._hidden_size, self._num_layers, batch_first=True)
        self._fc = nn.Linear(self._hidden_size, self._output_size)

    def _define_loss_criterion(self):
        return nn.MSELoss()

    def _initialise_optimiser(self):
        return optim.AdamW(
            self.parameters(),
            lr=self._training_parameters["learning_rate"],
            weight_decay=self._training_parameters["weight_decay"]
        )

    def _forward_pass(self, x):
        batch_size = x.size(0)

        h_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)

        output, h_0 = self._rnn(x, h_0.detach())
        output = output[:, -1, :]
        output = self._fc(output)

        return output

    @property
    def _model_save_name(self):
        return f"rnn_model_{int(time.time())}"
