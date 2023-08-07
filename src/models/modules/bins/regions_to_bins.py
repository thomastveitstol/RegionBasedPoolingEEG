"""
Classes for RBP
"""
import warnings
from itertools import cycle
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import enlighten
import random
import torch
import torch.nn as nn

from src.utils import ChGroup
from src.data.region_split import ChannelSplit
from src.data.datasets.data_base import BaseChannelSystem
from src.models.modules.pooling_modules.main_pooling_module import GroupPoolingModule
from src.models.modules.pooling_modules.univariate_rocket import UnivariateRocketKernels


# -------------------------------------------------
# Bin classes
# -------------------------------------------------
# Single bin
class _Bin(nn.Module):

    def __init__(self,
                 channel_split: ChannelSplit,
                 path: Union[Tuple[ChGroup, ...], List[ChGroup]],
                 pooling_method: str,
                 pooling_hyperparams: Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]):
        super().__init__()

        # --------------------------------
        # Define the channel split object
        # --------------------------------
        self._channel_split = channel_split

        # --------------------------------
        # Set the path
        # --------------------------------
        self._path = path

        # --------------------------------
        # Define pooling modules
        # --------------------------------
        self._pooling_modules = GroupPoolingModule(pooling_method=pooling_method,
                                                   num_pooling_modules=len(path),
                                                   hyperparameters=pooling_hyperparams)

    # ------------------------------------
    # Saving and loading
    # ------------------------------------
    def save(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_disk(cls, bin_state_dict: OrderedDict[str, torch.Tensor], channel_split: ChannelSplit,
                  path: Union[Tuple[ChGroup], List[ChGroup]], pooling_method: str,
                  pooling_hyperparams: Tuple[Dict[str, Any], ...]) -> '_Bin':
        """For loading previously saved model"""
        # Initialise object
        bin_model = cls(channel_split=channel_split, path=path, pooling_method=pooling_method,
                        pooling_hyperparams=pooling_hyperparams)

        # Load parameters
        bin_model.load_state_dict(state_dict=bin_state_dict, strict=True)
        return bin_model

    @classmethod
    def generate_module(
            cls,
            pooling_method: str,
            min_nodes: int,
            nodes: Dict[str, Tuple[float, float]],
            channel_systems: Union[BaseChannelSystem, List[BaseChannelSystem], Tuple[BaseChannelSystem, ...]],
            pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None,
            candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
            channel_split_hyperparams: Optional[Dict[str, Any]] = None) -> '_Bin':
        """
        Generate a Self object

        (unittest in test folder)
        Args:
            pooling_method: Name of Group Pooling module
            min_nodes: Stopping criteria for splitting the EEG cap into regions
            nodes: Nodes/electrodes to fit the split-algorithm on
            pooling_hyperparams: Hyperparameters of pooling modules
            channel_systems: Channel systems to fit. The channel systems passed are guaranteed to be compatible with the
                channel split
            candidate_region_splits: The candidate region splits
            channel_split_hyperparams: Hyperparameters of the channel split to make

        Returns:
            an object of type Self
        """
        # Convert None types to dict
        channel_split_hyperparams = dict() if channel_split_hyperparams is None else channel_split_hyperparams
        pooling_hyperparams = dict() if pooling_hyperparams is None else pooling_hyperparams

        # --------------------------------
        # Define the channel split object and fit it
        # --------------------------------
        channel_split = ChannelSplit(min_nodes=min_nodes, nodes=nodes, candidate_region_splits=candidate_region_splits,
                                     **channel_split_hyperparams)
        channel_split.fit_allowed_groups(channel_systems=channel_systems)
        channel_split.fit_channel_systems(channel_systems=channel_systems)

        # --------------------------------
        # Define paths within the requirements of
        # channel split object
        # --------------------------------
        allowed_groups = channel_split.allowed_ch_groups
        path = tuple(random.sample(channel_split.allowed_ch_groups, k=len(allowed_groups)))

        return cls(channel_split=channel_split, path=path, pooling_method=pooling_method,
                   pooling_hyperparams=pooling_hyperparams)

    # ------------------------------------
    # Forward methods and pre-computing
    # ------------------------------------
    def precompute(self, x: torch.Tensor, channel_system_name: str,
                   channel_name_to_index: Dict[str, int]) -> Optional[Tuple[torch.Tensor, ...]]:
        """Use pooling pre-computing for all groups in the bin"""
        # If pre-computing is not supported, just return None
        if not self._pooling_modules.supports_precomputing:
            return None

        channel_names = tuple(self._channel_split.channel_splits[channel_system_name][region] for region in self._path)
        return self._pooling_modules.precompute(x=x, ch_names=channel_names,
                                                channel_name_to_index=channel_name_to_index)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Extract a single bin from EEG data

        (unittest in test folder)
        Args:
            **kwargs: Input arguments needed to run forward pass. The necessary inputs depend on the selected pooling
                modules, and may include:
                    x: Full EEG data
                    channel_system_name: Name of the channel system
                    channel_name_to_index: Dict which maps from channel name to index
                    precomputed: Precomputed features. Should contain the features for all regions in the path
        Returns: A single bin
        """
        # Input  check
        legal_kwargs = "x", "precomputed", "ch_names", "channel_name_to_index"
        unused_args = tuple(forward_arg for forward_arg in self._pooling_modules.forward_args
                            if forward_arg not in legal_kwargs)
        if unused_args:
            warnings.warn(f"Expected all input keyword arguments to be in {legal_kwargs}, but found the following "
                          f"unsupported arguments: {unused_args}")

        # Initialise dictionary which will be input to the pooling modules
        inputs: Dict[str, Any] = dict()

        # Add necessary inputs to dict
        if "x" in self._pooling_modules.forward_args:
            inputs["x"] = kwargs["x"]
        if "precomputed" in self._pooling_modules.forward_args:
            inputs["precomputed"] = kwargs["precomputed"]
        if "ch_names" in self._pooling_modules.forward_args:
            channel_names = tuple(
                self._channel_split.channel_splits[kwargs["channel_system_name"]][region] for region in self._path)
            inputs["ch_names"] = channel_names
        if "channel_name_to_index" in self._pooling_modules.forward_args:
            inputs["channel_name_to_index"] = kwargs["channel_name_to_index"]

        return self._pooling_modules(**inputs)

    # ------------------------------------
    # Methods for fitting new channel systems
    # ------------------------------------
    def fit_channel_systems(self, channel_systems: Union[BaseChannelSystem, List[BaseChannelSystem],
                                                         Tuple[BaseChannelSystem, ...]]) -> None:
        """Fits multiple channel systems"""
        self._channel_split.fit_channel_systems(channel_systems=channel_systems)

    # ------------------------------------
    # Properties
    # ------------------------------------
    @property
    def path(self) -> Tuple[ChGroup, ...]:
        return self._path

    @path.setter
    def path(self, value: Tuple[ChGroup, ...]) -> None:
        self._path = value

    @property
    def channel_split(self) -> ChannelSplit:
        return self._channel_split

    @channel_split.setter
    def channel_split(self, value: ChannelSplit) -> None:
        self._channel_split = value

    @property
    def pooling_method(self) -> str:
        """Get the name of the group pooling module used"""
        return self._pooling_modules.name

    @property
    def pooling_hyperparams(self) -> Dict[str, Any]:
        """Get the hyperparameters of the pooling module"""
        return self._pooling_modules.hyperparameters

    @property
    def supports_precomputing(self) -> bool:
        return self._pooling_modules.supports_precomputing


class Regions2Bins(nn.Module):

    def __init__(self,
                 channel_splits: Tuple[ChannelSplit, ...],
                 paths: Tuple[Tuple[ChGroup, ...], ...],
                 pooling_methods: Union[str, Tuple[str, ...]],
                 pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...],
                                                     Tuple[Tuple[Dict[str, Any], ...], ...]]] = None):
        """
        Initialise. But better to use generate_module

        Args:
            channel_splits: Channel splits of the bins
            paths: Paths of the bins
            pooling_methods: The pooling methods to use. If str, the same pooling method is used for all channel splits
            pooling_hyperparams: Hyperparameters of the pooling modules
        """
        super().__init__()

        # Input check and conversion
        if isinstance(pooling_methods, str):
            pooling_methods = (pooling_methods,)

        pooling_hyperparams = (dict(),) if pooling_hyperparams is None else \
            pooling_hyperparams
        if isinstance(pooling_hyperparams, dict):
            pooling_hyperparams = (pooling_hyperparams,)

        # -------------------------------
        # Create bins
        # -------------------------------
        self._bins = nn.ModuleList(
            [_Bin(channel_split=channel_split, path=path, pooling_method=name,
                  pooling_hyperparams=pooling_module_hyperparams)
             for channel_split, path, name, pooling_module_hyperparams in zip(channel_splits, paths,
                                                                              cycle(pooling_methods),
                                                                              cycle(pooling_hyperparams))]
        )

    @classmethod
    def generate_module(cls,
                        channel_systems: Union[BaseChannelSystem, Tuple[BaseChannelSystem, ...]],
                        nodes: Dict[str, Tuple[float, float]],
                        num_channel_splits: int,
                        pooling_methods: Union[str, Tuple[str, ...]],
                        candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
                        min_nodes: int = 1,
                        channel_split_hyperparams: Optional[Dict[str, Any]] = None,
                        pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None
                        ) -> 'Regions2Bins':

        # Convert None types to dict
        channel_split_hyperparams = dict() if channel_split_hyperparams is None else channel_split_hyperparams

        # -----------------
        # Create channel splits
        # -----------------
        channel_splits = ChannelSplit.generate_multiple_channel_splits(
            num_channel_splits=num_channel_splits, nodes=nodes,
            min_nodes=min_nodes,
            candidate_region_splits=candidate_region_splits, **channel_split_hyperparams)
        paths: List[Tuple[ChGroup, ...]] = []

        # -----------------
        # Fit channel splits
        # -----------------
        for channel_split in channel_splits:
            channel_split.fit_allowed_groups(channel_systems=channel_systems)
            channel_split.fit_channel_systems(channel_systems=channel_systems)

            # --------------------------------
            # Define paths within the requirements of
            # channel split object
            # --------------------------------
            allowed_groups = channel_split.allowed_ch_groups
            paths.append(tuple(random.sample(channel_split.allowed_ch_groups, k=len(allowed_groups))))

        return cls(channel_splits=channel_splits, paths=tuple(paths),
                   pooling_methods=pooling_methods,
                   pooling_hyperparams=pooling_hyperparams)

    # ------------------------------------
    # Loading and saving
    # ------------------------------------
    def save(self, path: str) -> None:
        """
        Save the Self object to path

        (unittests in test folder)
        Args:
            path: path of which to save the Self object to

        Returns: Nothing
        """
        # Get state (everything needed to load the model)
        state = {"state_dict": self.state_dict(), "paths": self.paths, "pooling_hyperparams": self.pooling_hyperparams,
                 "pooling_methods": self.pooling_methods, "channel_splits": self.channel_splits}

        # Save
        torch.save(state, f"{path}")

    @classmethod
    def from_disk(cls, path: str) -> 'Regions2Bins':
        """
        Method for loading previously saved model

        (unittests in test folder)
        Args:
            path: Path of the where the object to load is stored

        Returns: An object of type Self, which is identical to the saved object
        """
        # Get state
        state = torch.load(path)

        # Initialise model
        model = cls(channel_splits=state["channel_splits"], paths=state["paths"],
                    pooling_methods=state["pooling_methods"], pooling_hyperparams=state["pooling_hyperparams"])

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model

    # ------------------------------------
    # Methods for fitting new channel systems
    # ------------------------------------
    def fit_channel_systems(self, channel_systems: Union[BaseChannelSystem, List[BaseChannelSystem],
                                                         Tuple[BaseChannelSystem, ...]]) -> None:
        """Fit channel systems on the channel split objects of the bins"""
        for bin_ in self._bins:
            bin_.fit_channel_systems(channel_systems=channel_systems)

    # ------------------------------------
    # Forward methods and pre-computing
    # ------------------------------------
    def pre_compute_batch(self, x: torch.Tensor, channel_system_name: str,
                          channel_name_to_index: Dict[str, int], batch_size: int, to_cpu: bool = True) \
            -> Optional[Tuple[Tuple[torch.Tensor, ...], ...]]:
        """
        Method for pre-computing features in batches
        Args:
            x: Full EEG data, with shape=(num_subjects, num_channels, num_time_steps)
            channel_system_name: Name of channel system. The bins must be fit on the channel system before using this
                method
            channel_name_to_index: Channel name to index (ChannelSystem specific)
            batch_size: Batch size
            to_cpu: To send the precomputed features to cpu after computing (True) or not (False)

        Returns:
            Returns a tuple where an element in the tuple is a bin. A bin is a tuple of torch tensors. The length of the
            output is the number of bins. The length of a bin is the number of slots in it (path length of sub-graph).
        """
        # Initialise list. The pre-computed features will be appended to this list, before it is returned as a tuple
        data: List[Tuple[torch.Tensor], ...] = []

        # ----------------------------
        # Loop though all bins
        #
        # It is necessary to loop through the bins
        # first and the subjects second to properly
        # handle the batch dimension
        # ----------------------------
        for bin_ in self._bins:
            # pre_computed_bin will be the complete pre-computed data of the current bin. That is, contain the
            # region-based pre-computed features/time series for all subjects in x
            pre_computed_bin = None  # This assignment is not strictly necessary, but PyCharm is confused otherwise

            # Loop through all subjects to store in the current bin
            for i in range(0, x.size()[0], batch_size):
                if i == 0:
                    # Initialise upon first batch. This is because the output size of the pre_compute method of the
                    # bins were unknown
                    first_batch_regions = bin_.precompute(x=x[i:(i + batch_size)],
                                                          channel_system_name=channel_system_name,
                                                          channel_name_to_index=channel_name_to_index)
                    # Initialise.
                    pre_computed_bin = [
                        torch.zeros(size=_get_size(num_subjects=x.size()[0], reminder=region.size()[1:])).to(x.device)
                        for region in first_batch_regions
                    ]
                    for main_region, subject_region in zip(pre_computed_bin, first_batch_regions):
                        # We want to store the pre-computed features of the current subject to the complete list
                        main_region[i:(i + batch_size)] = torch.squeeze(subject_region, dim=0)
                else:
                    # pre-compute the regions-based features/time series of the i-th subject
                    subject_regions = bin_.precompute(x=x[i:(i + batch_size)],
                                                      channel_system_name=channel_system_name,
                                                      channel_name_to_index=channel_name_to_index)

                    # Store the pre-computed regions features in the i-th position of the complete list
                    for main_region, subject_region in zip(pre_computed_bin, subject_regions):
                        main_region[i:(i + batch_size)] = subject_region

            # Maybe send to cpu
            if to_cpu:
                pre_computed_bin = _precomputed_bin_to_cpu(precomputed_bin=pre_computed_bin)

            # Convert to tuple and append to complete bins data
            data.append(tuple(pre_computed_bin))

        # After looping through all bins, convert to tuple and return
        return tuple(data)

    def pre_compute_full_dataset(self, x: torch.Tensor, channel_system_name: str,
                                 channel_name_to_index: Dict[str, int]) \
            -> Optional[Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Method for precomputing without batches (entire dataset at once). It may be infeasible for memory reasons"""
        rocket_features = tuple(bin_.precompute(x=x, channel_system_name=channel_system_name,
                                                channel_name_to_index=channel_name_to_index) for bin_ in self._bins)
        return rocket_features

    def forward(self,
                precomputed: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
                **inputs) -> Tuple[torch.Tensor, ...]:
        """Forward method. It just loops through all the bins, and returns as a tuple with length=num_bins. (Unit test
        in test folder)"""
        precomputed = cycle([None]) if precomputed is None else precomputed
        return tuple([bin_(precomputed=pre_comp, **inputs) for bin_, pre_comp in zip(self._bins, precomputed)])

    # ---------------
    # Properties
    # ---------------
    @property
    def paths(self) -> Tuple[Tuple[ChGroup, ...], ...]:
        return tuple([bin_.path for bin_ in self._bins])

    @paths.setter
    def paths(self, values: Tuple[Tuple[ChGroup, ...], ...]) -> None:
        """Set the paths in all bins"""
        for bin_, value in zip(self._bins, values):
            bin_.path = value

    @property
    def pooling_hyperparams(self) -> Tuple[Dict[str, Any], ...]:
        """Get the hyperparameters of the pooling modules"""
        return tuple(bin_.pooling_hyperparams for bin_ in self._bins)

    @property
    def pooling_methods(self) -> Tuple[str, ...]:
        return tuple(bin_.pooling_method for bin_ in self._bins)

    @property
    def channel_splits(self) -> Tuple[ChannelSplit, ...]:
        return tuple([bin_.channel_split for bin_ in self._bins])

    @channel_splits.setter
    def channel_splits(self, values: Tuple[ChannelSplit, ...]) -> None:
        """Set channel splits of the bins"""
        for bin_, value in zip(self._bins, values):
            bin_.channel_split = value

    @property
    def supports_precomputing(self) -> bool:
        return any(bin_.supports_precomputing for bin_ in self._bins)


class SharedPrecomputingRegions2Bins(nn.Module):
    """
    Same as Regions2Bins, but the precomputing is per channel, not per channel split
    """

    def __init__(self,
                 channel_splits: Tuple[ChannelSplit, ...],
                 paths: Tuple[Tuple[ChGroup, ...], ...],
                 rocket_implementation: Optional[int] = None,
                 num_kernels: Optional[int] = None,
                 max_receptive_field: Optional[int] = None,
                 pooling_method: str = "SharedRocketKernels",
                 pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...],
                                                     Tuple[Tuple[Dict[str, Any], ...], ...]]] = None):
        super().__init__()
        # Input check
        if isinstance(pooling_method, tuple):
            if len(set(pooling_method)) != 1:
                raise ValueError("Cannot have multiple (different) pooling mechanisms for this class")

            # If all pooling methods are the sam, it is ok
            pooling_method = pooling_method[0]

        # -----------------------
        # Define ROCKET kernels. Only a single rocket feature
        # extractor is used
        # -----------------------
        self._rocket_kernels = UnivariateRocketKernels(rocket_implementation=rocket_implementation,
                                                       num_kernels=num_kernels, max_receptive_field=max_receptive_field)

        # -----------------------
        # Create bins. They will use shared rocket kernels
        # -----------------------
        pooling_hyperparams = (dict(),) if pooling_hyperparams is None else pooling_hyperparams
        if isinstance(pooling_hyperparams, dict):
            pooling_hyperparams = (pooling_hyperparams,)

        for hyperparams in pooling_hyperparams:
            hyperparams["in_features"] = 2 * self._rocket_kernels.hyperparameters["num_kernels"]

        self._bins = nn.ModuleList(
            [_Bin(channel_split=channel_split, path=path, pooling_method=pooling_method,
                  pooling_hyperparams=pooling_module_hyperparams)
             for channel_split, path, pooling_module_hyperparams in zip(channel_splits, paths,
                                                                        cycle(pooling_hyperparams))]
        )

    @classmethod
    def generate_module(cls,
                        channel_systems: Union[BaseChannelSystem, List[BaseChannelSystem],
                                               Tuple[BaseChannelSystem, ...]],
                        nodes: Dict[str, Tuple[float, float]],
                        num_channel_splits: int,
                        candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
                        min_nodes: int = 1,
                        rocket_implementation: Optional[int] = None,
                        num_kernels: Optional[int] = None,
                        max_receptive_field: Optional[int] = None,
                        channel_split_hyperparams: Optional[Dict[str, Any]] = None,
                        pooling_method: str = "SharedRocketKernels",
                        pooling_hyperparams: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None
                        ) -> 'SharedPrecomputingRegions2Bins':
        # Convert None types to dict
        channel_split_hyperparams = dict() if channel_split_hyperparams is None else channel_split_hyperparams

        # -----------------
        # Create channel splits
        # -----------------
        channel_splits = ChannelSplit.generate_multiple_channel_splits(
            num_channel_splits=num_channel_splits, nodes=nodes, min_nodes=min_nodes,
            candidate_region_splits=candidate_region_splits, **channel_split_hyperparams)
        paths: List[Tuple[ChGroup, ...]] = []

        # -----------------
        # Fit channel splits
        # -----------------
        for channel_split in channel_splits:
            channel_split.fit_allowed_groups(channel_systems=channel_systems)
            channel_split.fit_channel_systems(channel_systems=channel_systems)

            # --------------------------------
            # Define paths within the requirements of
            # channel split object
            # --------------------------------
            allowed_groups = channel_split.allowed_ch_groups
            paths.append(tuple(random.sample(channel_split.allowed_ch_groups, k=len(allowed_groups))))

        return cls(channel_splits=channel_splits, paths=tuple(paths),
                   pooling_hyperparams=pooling_hyperparams,
                   pooling_method=pooling_method,
                   rocket_implementation=rocket_implementation, num_kernels=num_kernels,
                   max_receptive_field=max_receptive_field)

    # ------------------------------------
    # Forward methods and pre-computing
    # ------------------------------------
    def precompute(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Initialise the tensor
        num_subjects, num_channels = x.size()[:2]
        num_kernels = self._rocket_kernels.hyperparameters["num_kernels"]  # consider making num kernels a property on
        # its own

        rocket_features = torch.zeros(size=(num_subjects, num_channels, 2*num_kernels))

        # Loop through in batches
        pbar = enlighten.Counter(total=num_subjects//batch_size, desc="Pre-computing", unit="batch")
        for i in range(0, num_subjects, batch_size):
            # Precompute the batch and send to cpu
            batch_features = self._rocket_kernels.precompute(x[i:(i+batch_size)]).to(torch.device("cpu"))

            # Store it in the main rocket feature tensor
            rocket_features[i:(i+batch_size)] = batch_features

            pbar.update()

        # Return after completion
        return rocket_features

    def forward(self, x: torch.Tensor, precomputed: torch.Tensor, channel_name_to_index: Dict[str, int],
                channel_system_name: str) -> Tuple[torch.Tensor, ...]:
        return tuple(bin_(x=x, precomputed=precomputed, channel_name_to_index=channel_name_to_index,
                          channel_system_name=channel_system_name) for bin_ in self._bins)

    # ------------------------------------
    # Loading and saving
    # ------------------------------------
    def save(self, path: str) -> None:
        """
        Save the Self object to path

        (unittests in test folder)
        Args:
            path: path of which to save the Self object to

        Returns: Nothing
        """
        # Get state (everything needed to load the model)
        state = {"state_dict": self.state_dict(), "paths": self.paths, "pooling_hyperparams": self.pooling_hyperparams,
                 "pooling_methods": self.pooling_methods, "channel_splits": self.channel_splits,
                 "rocket_hyperparams": self._rocket_kernels.inputs}
        # Save
        torch.save(state, f"{path}")

    @classmethod
    def from_disk(cls, path: str) -> 'SharedPrecomputingRegions2Bins':
        """
        Method for loading previously saved model

        (unittests in test folder)
        Args:
            path: Path of the where the object to load is stored

        Returns: An object of type Self, which is identical to the saved object
        """
        # Get state
        state = torch.load(path)

        # Initialise model
        model = cls(channel_splits=state["channel_splits"], paths=state["paths"],
                    pooling_hyperparams=state["pooling_hyperparams"], pooling_method=state["pooling_methods"],
                    **state["rocket_hyperparams"])

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model

    # ---------------
    # Properties
    # ---------------
    @property
    def supports_precomputing(self) -> bool:
        return any(bin_.supports_precomputing for bin_ in self._bins)

    @property
    def paths(self) -> Tuple[Tuple[ChGroup, ...], ...]:
        return tuple([bin_.path for bin_ in self._bins])

    @paths.setter
    def paths(self, values: Tuple[Tuple[ChGroup, ...], ...]) -> None:
        """Set the paths in all bins"""
        for bin_, value in zip(self._bins, values):
            bin_.path = value

    @property
    def pooling_hyperparams(self) -> Tuple[Dict[str, Any], ...]:
        """Get the hyperparameters of the pooling modules"""
        return tuple(bin_.pooling_hyperparams for bin_ in self._bins)

    @property
    def rocket_hyperparams(self) -> Dict[str, Any]:
        return self._rocket_kernels.inputs

    @property
    def pooling_methods(self) -> Tuple[str, ...]:
        return tuple(bin_.pooling_method for bin_ in self._bins)

    @property
    def channel_splits(self) -> Tuple[ChannelSplit, ...]:
        return tuple([bin_.channel_split for bin_ in self._bins])

    @channel_splits.setter
    def channel_splits(self, values: Tuple[ChannelSplit, ...]) -> None:
        """Set channel splits of the bins"""
        for bin_, value in zip(self._bins, values):
            bin_.channel_split = value


# -------------------------------------------------
# Functions
# -------------------------------------------------
def _get_size(num_subjects: int, reminder: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    """
    This is to be used for obtaining the size of a tensor by unpacking, when it is most convenient to not only pass
    integers
    Args:
        num_subjects: Number of subjects
        reminder: Either a tuple of ints or a torch.Size object. This will be "unpacked" such that a tuple of ints
            is obtained

    Returns:
        Tuple of ints, where the first element is num_subjects and the remaining are the elements of remainder

    Examples:
        >>> _get_size(num_subjects=743, reminder=(987, 639, 83))
        (743, 987, 639, 83)
        >>> _get_size(num_subjects=74, reminder=torch.rand(size=(43, 12, 87, 4)).size())
        (74, 43, 12, 87, 4)
    """
    # Unpack the tuple
    shape = [size for size in reminder]

    # Insert num subject at front
    shape.insert(0, num_subjects)

    # return as tuple
    return tuple(shape)


def _precomputed_bin_to_cpu(precomputed_bin: List[torch.Tensor]) -> List[torch.Tensor]:
    """Function for sending a list of torch tensors to cpu"""
    return [precomputed.to("cpu") for precomputed in precomputed_bin]
