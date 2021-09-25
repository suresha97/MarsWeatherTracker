from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import json
import time

import sklearn.metrics as metrics
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import Tensor

_MODEL_SAVE_PATH = "../trained_models"


class TorchTrainer(ABC, nn.Module):
    def __init__(
            self,
            training_parameters: Dict[str, Any],
            forecasting_data: Dict[str, Tuple[Tensor, Tensor]],
            quantity_to_forecast: str
    ) -> None:
        super().__init__()
        super(ABC, self).__init__()
        self._training_parameters = training_parameters
        self._forecasting_data = forecasting_data
        self._train_loader = self._get_torch_data_loader(self._training_parameters["batch_size"])
        self._quantity_to_forecast = quantity_to_forecast

    def _get_torch_data_loader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        train_features, train_labels = self._forecasting_data["train_data"]
        dataset = TensorDataset(train_features, train_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def train_model(self) -> None:
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
                model_preds = self._model.forward(train_in)

                loss = loss_criterion(model_preds, train_out)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

            training_loss_over_epochs.append(running_loss / len(self._train_loader))

            with torch.no_grad():
                self._model.eval()

                val_preds = self._model.forward(val_features)
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

    def predict(self, prediction_features: Tensor) -> Tensor:
        self._model.eval()
        return self._model.forward(prediction_features)

    def evaluate_model(self) -> None:
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

    def save_model(self) -> None:
        torch.save(
            self._model.state_dict(), f"{_MODEL_SAVE_PATH}/{self._quantity_to_forecast}/"
                                      f"{self._model_save_name}_{self._save_time}.pt"
        )

    def save_model_architecture(self) -> None:
        with open(
                f"{_MODEL_SAVE_PATH}/{self._quantity_to_forecast}/"
                f"{self._model_save_name}_architecture_inputs_{self._save_time}.json", "w"
        ) as f:
            json.dump(self._model_architecture_inputs, f)

    @property
    def _save_time(self) -> int:
        return int(time.time())

    @abstractmethod
    def _define_loss_criterion(self) -> None:
        return

    @abstractmethod
    def _initialise_optimiser(self) -> None:
        return

    @property
    @abstractmethod
    def _model_save_name(self) -> None:
        return
