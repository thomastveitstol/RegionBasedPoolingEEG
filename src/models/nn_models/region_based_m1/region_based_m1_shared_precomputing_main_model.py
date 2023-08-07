"""
Main model, which integrates all submodules together
"""
from collections import OrderedDict
import copy
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import enlighten
import torch
import torch.nn as nn
from torch import optim
from torch.nn.modules.loss import _Loss  # noqa
from torch.utils.data import DataLoader

from src.data.region_split import ChannelSplit
from src.data.datasets.data_base import BaseChannelSystem
from src.metrics import Histories
from src.models.modules.classifiers.main_mts_classifier import MTSClassifier
from src.models.nn_models.main_model_base import MainModelBase
from src.models.modules.bins.regions_to_bins import SharedPrecomputingRegions2Bins
from src.utils import ChGroup


class SharedRocketClassifierM1(MainModelBase):
    """
    Similar to Regions2BinsClassifierM1 with ROCKET features for computing coefficients, but the ROCKET kernels are
    shared
    """
    def __init__(self,
                 channel_splits: Tuple[ChannelSplit, ...],
                 paths: Tuple[Tuple[ChGroup], ...],
                 mts_classifier_name: str,
                 mts_classifier_hyperparams: Dict[str, Any],
                 pooling_method: str,
                 pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...],
                                                     Tuple[Tuple[Dict[str, Any], ...], ...]]] = None,
                 normalise_region_representations: bool = True,
                 rocket_implementation: Optional[int] = None,
                 num_kernels: Optional[int] = None,
                 max_receptive_field: Optional[int] = None,
                 batch_norm: bool = False,
                 ) -> None:
        super().__init__()

        # ---------------------------------------
        # Define module for selecting region-paths
        # and inserting them in bins
        # ---------------------------------------
        self._regions_2_bins_module = SharedPrecomputingRegions2Bins(
            channel_splits=channel_splits, paths=paths, rocket_implementation=rocket_implementation,
            num_kernels=num_kernels, max_receptive_field=max_receptive_field,
            pooling_method=pooling_method,
            pooling_hyperparams=pooling_hyperparams)

        self._normalise_region_representations = normalise_region_representations  # TODO: this is not saved
        self._batch_norm = nn.BatchNorm1d(mts_classifier_hyperparams["in_channels"]) if batch_norm else None

        # ---------------------------------------
        # Define module for going from MTS to
        # classification
        # ---------------------------------------
        self._bins_classifier = MTSClassifier(classifier_name=mts_classifier_name,
                                              **mts_classifier_hyperparams)

    @classmethod
    def generate_model(cls,
                       stacked_bins_classifier_name: str,
                       stacked_bins_classifier_hyperparams: Dict[str, Any],
                       nodes: Dict[str, Tuple[float, float]],
                       num_channel_splits: int,
                       min_nodes: int = 1,
                       candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
                       pooling_method: str = "SharedRocketKernels",
                       pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None,
                       channel_split_hyperparams: Optional[Dict[str, Any]] = None,
                       channel_systems: Optional[Union[BaseChannelSystem, List[BaseChannelSystem],
                                                       Tuple[BaseChannelSystem, ...]]] = None,
                       normalise_region_representations: bool = True,
                       rocket_implementation: Optional[int] = None,
                       num_kernels: Optional[int] = None,
                       max_receptive_field: Optional[int] = None,
                       batch_norm: bool = False,
                       ):
        """
        Method for generating a new model randomly

        (unit test in test folder)
        Args:
            max_receptive_field: Max receptive field of the ROCKET kernels
            num_kernels: Number of ROCKET kernels
            rocket_implementation: Rocket implementation to use
            candidate_region_splits: Candidates of region splits
            min_nodes: Minimum number of nodes allowed in a single region
            num_channel_splits: Number of channel splits
            nodes: Dictionary of electrode positions. The keys are channel names, the values are 2D positions
            channel_split_hyperparams: Hyperparameters of the channel split
            stacked_bins_classifier_name: name of stacked bins classifier
            stacked_bins_classifier_hyperparams: hyperparameters of stacked bins classifier
            pooling_method: Name of group pooling module
            pooling_hyperparams: hyperparameters of pooling module
            channel_systems: channels systems to fit
            normalise_region_representations: To normalise the region representations before they are passed to the
                classifier.
            batch_norm: To use batch norm after region based pooling (True) or not (False)

        Returns:
            an object of type Self
        """
        # ---------------------------------------
        # Define module for selecting region-paths
        # and inserting them in bins
        # ---------------------------------------
        channel_split_hyperparams = dict() if channel_split_hyperparams is None else channel_split_hyperparams

        # Note that 'max_receptive_field' and 'num_kernels' are not actually used here. However, they are still required
        # keyword arguments, so we just pass in something random
        regions_2_bins_module = SharedPrecomputingRegions2Bins.generate_module(
            num_channel_splits=num_channel_splits, channel_systems=channel_systems, min_nodes=min_nodes,
            channel_split_hyperparams=channel_split_hyperparams, candidate_region_splits=candidate_region_splits,
            pooling_hyperparams=pooling_hyperparams, max_receptive_field=40, num_kernels=10,
            pooling_method=pooling_method, rocket_implementation=rocket_implementation, nodes=nodes)

        # ---------------------------------------
        # Define module for going from bins to
        # classification
        # ---------------------------------------
        slots = [len(path) for path in regions_2_bins_module.paths]
        stacked_bins_classifier_hyperparams["in_channels"] = sum(slots)
        bins_classifier = MTSClassifier(classifier_name=stacked_bins_classifier_name,
                                        **stacked_bins_classifier_hyperparams)

        return cls(channel_splits=regions_2_bins_module.channel_splits, paths=regions_2_bins_module.paths,
                   rocket_implementation=rocket_implementation, num_kernels=num_kernels,
                   max_receptive_field=max_receptive_field, mts_classifier_name=stacked_bins_classifier_name,
                   mts_classifier_hyperparams=bins_classifier.hyperparameters, pooling_method=pooling_method,
                   pooling_hyperparams=regions_2_bins_module.pooling_hyperparams,
                   normalise_region_representations=normalise_region_representations, batch_norm=batch_norm)

    def fit_channel_system(self, channel_system: BaseChannelSystem) -> None:
        """Fits the channel systems on the channel splits of the bins"""
        self._regions_2_bins_module.fit_channel_systems(channel_systems=channel_system)

    # -------------------------
    # Forward methods
    # -------------------------
    def pre_compute(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Method for pre-computing features"""
        return self._regions_2_bins_module.precompute(x=x, batch_size=batch_size)

    def forward(self,
                x: torch.Tensor,
                channel_system_name: str,
                channel_name_to_index: Dict[str, int],
                precomputed: Optional[Tuple[Tuple[torch.Tensor]]] = None) -> torch.Tensor:
        """
        The forward method. For predictions, use the predict method, as this supports calculating gradients and do not
        use sigmoid. This is because it is more numerically stable to not apply sigmoid, but rather use
        BCEWithLogitsLoss instead of BCELoss

        (unit test in test folder)
        Args:
            x: EEG data, a torch tensor with shape=(batch, channels, time_steps). Once trained, the forward method
                can still be used on EEG data with different number of channels and time steps (and batch size,
                obviously)
            precomputed: Pre-computed features/time series
            channel_system_name: Name of channel system
            channel_name_to_index: See ChannelSystem
        Returns:
            Predictions, without applying sigmoid
        """
        # To region-representations (bins)
        x = self._regions_2_bins_module(x=x, precomputed=precomputed,
                                        channel_system_name=channel_system_name,
                                        channel_name_to_index=channel_name_to_index)
        x = torch.cat(tensors=x, dim=1)

        if self._normalise_region_representations:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)

        if self._batch_norm is not None:
            x = self._batch_norm(x)

        # Pass through classifier (typically without activation function)
        x = self._bins_classifier(x)

        return x

    def predict(self, x: torch.Tensor, channel_system_name: str, channel_name_to_index: Dict[str, int],
                precomputed: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None) -> torch.Tensor:
        """Used for predictions, as this automatically turns off gradient calculations and applies sigmoid activation
        function"""
        # TODO: This does not work properly unless num_classes is set to 1 in StackedBinsClassifier
        with torch.no_grad():
            predictions = torch.sigmoid(self(x=x, precomputed=precomputed,
                                             channel_system_name=channel_system_name,
                                             channel_name_to_index=channel_name_to_index))
        return predictions

    def test_model(self, *, loader: DataLoader, channel_systems: Tuple[BaseChannelSystem, ...],
                   histories: Dict[str, Histories], device: torch.device,
                   activation_function: Optional[Callable] = None) -> None:
        """
        Method for testing

        Args:
            loader: Data loader of data to test
            channel_systems: Channel systems to test on
            histories: Histories object to be updated
            device: Device for running computations
            activation_function: Activation function to use. It will only be used to compute metrics, not prior to
                computing loss

        Returns:
            Nothing. The Histories object is updated in-place
        """
        self.eval()
        with torch.no_grad():
            for data, targets, pre_computed in loader:
                x = data.to(device)
                y = targets.to(device)
                precomputed_features = pre_computed.to(device)

                # Compute validation metrics for all channel systems
                for channel_system in channel_systems:
                    # Predict
                    y_pred = self(x=x, precomputed=precomputed_features,
                                  channel_system_name=channel_system.name,
                                  channel_name_to_index=channel_system.channel_name_to_index())
                    if activation_function is not None:
                        y_pred = activation_function(y_pred)

                    # Store in correct Histories object
                    histories[channel_system.name].store_batch_evaluation(y_pred=y_pred, y_true=y)

            # Call .on_epoch_end() on all validation histories
            for history in histories.values():
                history.on_epoch_end()

    # -------------------------
    # Training
    # -------------------------
    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, train_history: Histories,
                  val_histories: Dict[str, Histories], train_channel_systems: Tuple[BaseChannelSystem, ...],
                  val_channel_systems: Tuple[BaseChannelSystem, ...], num_epochs: int, device: torch.device,
                  criterion: _Loss, optimiser: optim.Optimizer, activation_function: Optional[Callable] = None) \
            -> Tuple[Histories, Dict[str, Histories]]:
        """
        Method for training. The model parameters are changed in-place

        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            train_history: History object for training
            val_histories: History objects for validation. The keys are channel system name, which must match with
                val_channel_systems
            train_channel_systems: Channel systems for training. When training, a channel system from this tuple is
                randomly selected per batch.
            val_channel_systems: Channel systems for validation.
            num_epochs: Number of epochs
            device: device of which to run the calculations on
            criterion: Loss function object
            optimiser: Optimiser object
            activation_function: Activation function to use. It will only be used to compute metrics, not prior to
                computing loss

        Returns: A tuple containing the updated Histories. The first is the histories for training, the second is for
            validation (keys are the same as the keys passed as input)

        """
        # ------------------------
        # Fit model
        # ------------------------
        best_state = None
        record = 0
        for epoch in range(num_epochs):
            pbar = enlighten.Counter(total=int(len(train_loader.dataset) / train_loader.batch_size + 1),  # type: ignore
                                     desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            # ---------------------
            # Train
            # ---------------------
            self.train()
            for data, targets, pre_computed in train_loader:
                # Send to device
                x = data.to(device)
                y = targets.to(device)
                precomputed_features = pre_computed.to(device)

                # Select a channel system at random to use for training
                train_channel_system: BaseChannelSystem = random.choice(train_channel_systems)
                channel_name_to_index = train_channel_system.channel_name_to_index()

                # Forward pass
                scores = self(x, precomputed=precomputed_features, channel_system_name=train_channel_system.name,
                              channel_name_to_index=channel_name_to_index)

                # Compute loss
                loss = criterion(scores, y)

                # Clear some memory
                # noinspection PyUnusedLocal
                x = None

                # noinspection PyUnusedLocal
                precomputed_features = None

                # Do a training step
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()  # For memory

                # Update train history
                with torch.no_grad():
                    y_pred = torch.clone(scores)
                    if activation_function is not None:
                        y_pred = activation_function(y_pred)
                    train_history.store_batch_evaluation(y_pred=y_pred, y_true=y)

                # Update pbar
                pbar.update()

                # Clear some more memory
                # noinspection PyUnusedLocal
                y = None

            train_history.on_epoch_end()

            # ---------------------
            # Evaluate
            # ---------------------
            self.eval()
            with torch.no_grad():
                for val_data, val_targets, val_pre_computed in val_loader:
                    x_val = val_data.to(device)
                    y_val = val_targets.to(device)
                    val_precomputed_features = val_pre_computed.to(device)

                    # Compute validation metrics for all channel systems
                    for channel_system in val_channel_systems:
                        # Predict
                        y_pred = self(x=x_val, precomputed=val_precomputed_features,
                                      channel_system_name=channel_system.name,
                                      channel_name_to_index=channel_system.channel_name_to_index())
                        if activation_function is not None:
                            y_pred = activation_function(y_pred)

                        # Store in correct Histories object
                        val_histories[channel_system.name].store_batch_evaluation(y_pred=y_pred, y_true=y_val)

                # Clear some memory
                # noinspection PyUnusedLocal
                x_val = None

                # noinspection PyUnusedLocal
                y_val = None

                # noinspection PyUnusedLocal
                val_precomputed_features = None

                # Call .on_epoch_end() on all validation histories
                for history in val_histories.values():
                    history.on_epoch_end()

            # Maybe update the best model and the record
            newest_auc = sum(val_history.newest_metrics["auc"] for val_history in val_histories.values())
            if newest_auc > record:
                best_state = copy.deepcopy(OrderedDict({k: v.cpu() for k, v in self.state_dict().items()}))
                record = newest_auc

        # Set the parameters back to those of the best model
        self.load_state_dict(OrderedDict({k: v.to(device) for k, v in best_state.items()}))
        return train_history, val_histories

    # -------------------------
    # Properties
    # -------------------------
    @property
    def channel_splits(self) -> Tuple[ChannelSplit, ...]:
        return self._regions_2_bins_module.channel_splits

    @channel_splits.setter
    def channel_splits(self, values: Tuple[ChannelSplit, ...]) -> None:
        self._regions_2_bins_module.channel_splits = values

    @property
    def paths(self) -> Tuple[Tuple[ChGroup, ...], ...]:
        return self._regions_2_bins_module.paths

    @paths.setter
    def paths(self, values: Tuple[Tuple[ChGroup, ...], ...]) -> None:
        self._regions_2_bins_module.paths = values

    @property
    def supports_precomputing(self) -> bool:
        return self._regions_2_bins_module.supports_precomputing

    # -------------------------
    # Loading and saving
    # -------------------------
    def save(self, path: str) -> None:
        # --------------
        # Save regions to bins
        # --------------
        self._regions_2_bins_module.save(path=os.path.join(path, "regions_2_bins.pt"))

        # --------------
        # Save MTS classifier
        # --------------
        self._bins_classifier.save(path=os.path.join(path, "mts_classifier.pt"))

        # --------------
        # Save model state dict
        # --------------
        torch.save(self.state_dict(), os.path.join(path, "model_state_dict.pt"))

    @classmethod
    def from_disk(cls, path: str) -> 'SharedRocketClassifierM1':
        """Method for loading model from disk"""
        # Make objects
        regions_2_bins = SharedPrecomputingRegions2Bins.from_disk(path=os.path.join(path, "regions_2_bins.pt"))
        mts_classifier = MTSClassifier.from_disk(path=os.path.join(path, "mts_classifier.pt"))

        # ----------------------
        # Initialise model
        # ----------------------
        # assert False, regions_2_bins.pooling_hyperparams[0].keys()
        model = cls(channel_splits=regions_2_bins.channel_splits, pooling_method=regions_2_bins.pooling_methods[0],
                    paths=regions_2_bins.paths, pooling_hyperparams=regions_2_bins.pooling_hyperparams,
                    num_kernels=regions_2_bins.rocket_hyperparams["num_kernels"],
                    max_receptive_field=regions_2_bins.rocket_hyperparams["max_receptive_field"],
                    rocket_implementation=regions_2_bins.rocket_hyperparams["rocket_implementation"],
                    mts_classifier_name=mts_classifier.name, mts_classifier_hyperparams=mts_classifier.hyperparameters)

        # ----------------------
        # Load parameters
        # ----------------------
        model.load_state_dict(torch.load(os.path.join(path, "model_state_dict.pt"), map_location="cpu"), strict=True)

        return model
