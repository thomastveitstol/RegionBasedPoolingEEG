import copy
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

import enlighten
import torch
from torch import optim
import torch.nn as nn
from torch.nn.modules.loss import _Loss  # noqa
from torch.utils.data import DataLoader

from src.metrics import Histories
from src.models.modules.classifiers.main_mts_classifier import MTSClassifier
from src.models.nn_models.main_model_base import MainModelBase


class FixedChannelDimMainModel(MainModelBase):
    """
    Main class for classifiers
    """

    def __init__(self, classifier_name: str, **kwargs):
        """
        Initialise
        Args:
            classifier_name: Name of the Multivariate TS classifier (see Classifier for options)
            **kwargs: Hyperparameters of the selected Multivariate TS classifier. See the selected ones for further
                specifications on required and optional kwargs
        Examples:
            >>> my_model = FixedChannelDimMainModel(classifier_name="Inception", in_channels=40, num_classes=3)
            >>> type(my_model._classifier).__name__
            'MTSClassifier'
            >>> my_model._hyperparameters
            {'in_channels': 40, 'num_classes': 3, 'classifier_name': 'Inception'}
            >>> type(FixedChannelDimMainModel(classifier_name="EEGResNet", in_channels=40, num_classes=3,
            ...                               time_steps=257)._classifier).__name__
            'MTSClassifier'
            >>> # Raises KeyError if classifier is not found
            >>> _ = FixedChannelDimMainModel(classifier_name="ThisIsNotARealClassifier", in_channels=40)
            Traceback (most recent call last):
            ...
            KeyError: 'Classifier ThisIsNotARealClassifier was not recognised as an MTS classifier'
        """
        super().__init__()
        # ---------------------------
        # Select classifier
        # ---------------------------
        self._classifier = MTSClassifier(classifier_name=classifier_name, **kwargs)

        # ---------------------------
        # Store hyperparameters (needed)
        # for loading the model
        # ---------------------------
        hyperparameters = kwargs.copy()
        hyperparameters["classifier_name"] = classifier_name
        self._hyperparameters = hyperparameters

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward method. (see Classifier for tests)"""
        return self._classifier(x, **kwargs)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Can be used for binary classification (turns off gradient calculations and applies sigmoid)"""
        with torch.no_grad():
            predictions = torch.sigmoid(self(x))
        return predictions

    def test_model(self, *, loader: DataLoader, histories: Dict[str, Histories], device: torch.device,
                   activation_function: Optional[Callable] = None) -> None:
        """
        Method for testing

        Args:
            loader: Data loader of data to test on
            histories: Histories object. This also contains all channel systems to test on, in its keys
            device: Device for running computations
            activation_function: Activation function to use for the predictions

        Returns:
            Nothing, the Histories objects are updated in-place
        """
        self.eval()
        with torch.no_grad():
            for (_, targets), data_interpolated in loader:
                # Send targets to device
                y = targets.to(device)

                # Compute validation metrics for all reduced channel systems
                for channel_system, history in histories.items():
                    # Select data
                    x_interpolated = data_interpolated[channel_system].to(device)

                    # Forward pass
                    predictions = self(x_interpolated)

                    # Update history
                    y_pred = torch.clone(predictions)
                    if activation_function is not None:
                        y_pred = activation_function(y_pred)
                    history.store_batch_evaluation(y_pred=y_pred, y_true=y)

            # Call .on_epoch_end() on all history objects in val_history
            for history in histories.values():
                history.on_epoch_end()

    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, train_history: Histories,
                  val_histories: Dict[str, Histories], num_epochs: int, device: torch.device,
                  criterion: _Loss, optimiser: optim.Optimizer, activation_function: Optional[Callable] = None) \
            -> Tuple[Histories, Dict[str, Histories]]:
        """
        Method for training. The model parameters are changed in-place
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            train_history: History object for training
            val_histories: History objects for validation. The keys are desired names of Histories objects, such as
                channel system name
            num_epochs: Number of epochs
            device: device of which to run the calculations on
            criterion: Loss function object
            optimiser: Optimiser object
            activation_function: Activation function to use. It will only be used to compute metrics, not prior to
                computing loss

        Returns: A tuple containing the updated Histories. The first is the histories for training, the second is for
            validation (keys are the same names as passed as input)

        """
        # ------------------------
        # Fit model
        # ------------------------
        best_state = None
        record = 0
        for epoch in range(num_epochs):
            pbar = enlighten.Counter(total=int(len(train_loader) / train_loader.batch_size + 1),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ---------------------
            # Train
            # ---------------------
            self.train()
            for data, targets in train_loader:
                # Send data to device
                x = data.to(device)

                if isinstance(criterion, nn.NLLLoss):
                    targets = targets.type(torch.LongTensor)
                y = targets.to(device)

                # Forward pass. Using only the original channel system
                scores = self(x)

                # Compute loss
                if isinstance(criterion, nn.NLLLoss):
                    loss = criterion(scores, torch.squeeze(y, dim=-1))
                else:
                    loss = criterion(scores, y)

                # Do a training step
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Update train history
                with torch.no_grad():
                    y_pred = torch.clone(scores)
                    if activation_function is not None:
                        y_pred = activation_function(y_pred)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y)

                # Update pbar
                pbar.update()

            train_history.on_epoch_end()

            # ---------------------
            # Evaluate
            # ---------------------
            self.eval()
            with torch.no_grad():
                for (_, val_targets), val_data_interpolated in val_loader:
                    # Send targets to device
                    y_val = val_targets.to(device)

                    # Compute validation metrics for all reduced channel systems
                    for channel_system, history in val_histories.items():
                        # Select data
                        x_val_interpolated = val_data_interpolated[channel_system].to(device)

                        # Forward pass
                        predictions = self(x_val_interpolated)

                        # Update history
                        y_pred = torch.clone(predictions)
                        if activation_function is not None:
                            y_pred = activation_function(y_pred)
                        history.store_batch_evaluation(y_pred=y_pred, y_true=y_val)

                # Call .on_epoch_end() on all history objects in val_history
                for history in val_histories.values():
                    history.on_epoch_end()

            # Maybe update the best model and the record
            newest_auc = sum(val_history.newest_metrics["auc"] for val_history in val_histories.values())
            if newest_auc > record:
                best_state = copy.deepcopy(self.state_dict())
                record = newest_auc

        # Set the parameters back to those of the best model
        self.load_state_dict(best_state)
        return train_history, val_histories

    # -------------------------
    # Loading and saving
    # -------------------------
    def save(self, path: str) -> None:
        """
        Method for saving model

        (unittest in test folder)
        """
        # Add a '/' if it is not the final character in the path string
        if path[-1] != "/":
            path += "/"

        # Save
        torch.save(self.state_dict(), f"{path}model_dict.pt")
        with open(f"{path}hyperparameters.pkl", "wb") as f:
            pickle.dump(self._hyperparameters, f)

    @classmethod
    def from_disk(cls, path: str) -> 'FixedChannelDimMainModel':
        """
        Method for loading model

        (unittest in test folder)
        """
        # Add a '/' if it is not the final character in the path string
        if path[-1] != "/":
            path += "/"

        # ----------------------
        # Initialise model
        # ----------------------
        # Load hyperparameters from path
        with open(f"{path}hyperparameters.pkl", "rb") as f:
            model_hyperparameters: Dict[str, Any] = pickle.load(f)

        # Initialise
        model = cls(**model_hyperparameters)

        # ----------------------
        # Load the parameters of the model
        # ----------------------
        model.load_state_dict(torch.load(f"{path}model_dict.pt"))

        return model

    # -------------
    # Properties
    # -------------
    @property
    def final_activation(self) -> str:
        return self._classifier.final_activation
