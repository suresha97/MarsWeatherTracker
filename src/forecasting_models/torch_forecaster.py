from abc import ABC, abstractmethod

import sklearn.metrics as metrics
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

_MODEL_SAVE_PATH = "../trained_models"


class TorchForecaster(ABC, nn.Module):
    def __init__(self, training_parameters, forecasting_data):
        super().__init__()
        super(ABC, self).__init__()
        self._training_parameters = training_parameters
        self._forecasting_data = forecasting_data
        self._train_loader = self._get_torch_data_loader(self._training_parameters["batch_size"])

    def _get_torch_data_loader(self, batch_size, shuffle=False):
        train_features, train_labels = self._forecasting_data["train_data"]
        dataset = TensorDataset(train_features, train_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def train_model(self):
        training_loss_over_epochs = []
        validation_loss_over_epochs = []

        val_features, val_labels = self._forecasting_data["validation_data"]
        num_epochs = self._training_parameters["num_epochs"]

        loss_criterion = self._define_loss_criterion()
        optimiser = self._initialise_optimiser()

        best_val_loss = 1000000

        for epoch in range(num_epochs):

            running_loss = 0
            for batch, (train_in, train_out) in enumerate(self._train_loader):
                model_preds = self._forward_pass(train_in)

                loss = loss_criterion(model_preds, train_out)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

            training_loss_over_epochs.append(running_loss / len(self._train_loader))

            with torch.no_grad():
                self.eval()

                val_preds = self._forward_pass(val_features)
                val_loss = loss_criterion(val_preds, val_labels)
                validation_loss_over_epochs.append(val_loss.item())

            print(f'Epoch [{epoch + 1}/{num_epochs}]], Loss: {loss.item():.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            if epoch - best_epoch > self._training_parameters["early_stopping_num_epochs"]:
                print(
                    f"Training stopped after {epoch} epochs. Training loss: {training_loss_over_epochs[-1]},"
                    f" Validaton loss: {val_loss}"
                )
                break

    def predict(self, prediction_features):
        self.eval()
        return self._forward_pass(prediction_features)

    def evaluate_model(self):
        test_features, test_labels = self._forecasting_data["test_data"]
        test_set_predictions = self.predict(test_features).detach().numpy()

        mae = metrics.mean_absolute_error(test_labels.numpy(), test_set_predictions)
        mse = metrics.mean_squared_error(test_labels.numpy(), test_set_predictions)
        mape = metrics.mean_absolute_percentage_error(test_labels.numpy(), test_set_predictions)

        print(
            {
                "mae": mae,
                "mse": mse,
                "mape": mape
            }
        )

    def save_model(self):
        torch.save(self.state_dict(), f"{_MODEL_SAVE_PATH}/{self._model_save_name}.pt")

    @abstractmethod
    def _define_loss_criterion(self):
        return

    @abstractmethod
    def _initialise_optimiser(self):
        return

    @abstractmethod
    def _forward_pass(self, x):
        return

    @property
    @abstractmethod
    def _model_save_name(self):
        return
