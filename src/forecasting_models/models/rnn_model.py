import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, model_architecture):
        super(RNNModel, self).__init__()
        self.input_size = model_architecture["input_size"]
        self.output_size = model_architecture["output_size"]
        self.hidden_size = model_architecture["hidden_size"]
        self.n_layers = model_architecture["num_layers"]
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        output, h_0 = self.rnn(x, h_0.detach())

        output = output[:, -1, :]

        output = self.fc(output)

        return output
