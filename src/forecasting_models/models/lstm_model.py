from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


class LSTMModel(nn.Module):
    def __init__(self, model_architecture: Dict[str, Any]) -> None:
        super(LSTMModel, self).__init__()
        self.input_size = model_architecture["input_size"]
        self.output_size = model_architecture["output_size"]
        self.hidden_size = model_architecture["hidden_size"]
        self.n_layers = model_architecture["num_layers"]
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        output, (h_0, c_0) = self.lstm(x, (h_0.detach(), c_0.detach()))

        output = output[:, -1, :]

        output = self.fc(output)

        return output
