from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn

from src.data.datasets.data_base import channel_names_to_indices
from src.models.customisations.rocket_convolution import RocketConv, RocketConv1d, RocketConv2d, compute_ppv_and_max
from src.models.modules.common_blocks.fc_blocks import FCBlock
from src.models.modules.pooling_modules.pooling_base import PoolingBase, GroupPoolingBase


class UnivariateRocket(PoolingBase):
    """
    Applies Random Convolutional Kernels on all time series in a univariate way (one row at a time). Rocket-like
    features are then extracted, and an FC block is applied with softmax as its final activation function
    """

    supports_precomputing = True
    forward_args = "x", "precomputed"

    def __init__(self,
                 rocket_implementation: int = 3,
                 num_kernels: int = 1000,
                 max_receptive_field: int = 500,
                 fc_block_id: int = 1,
                 fc_units: Optional[Tuple[int, ...]] = None,
                 num_region_representations: int = 1):
        """
        Initialise

        (unittests in test folder)
        Args:
            rocket_implementation: Implementation of ROCKET feature extractor to use. They should be mathematically ,
                equivalent, but have different pros/cons with respect to memory usage and efficiency
            num_kernels: Number of random convolutional kernels used
            max_receptive_field: Max receptive field (upper bound) of the kernels
            fc_block_id: ID of the FC-module used
            fc_units: Units of the FC-module
        """
        super().__init__()

        # Maybe set default
        fc_units = [num_kernels // 2] if fc_units is None else list(fc_units)
        self.update_input_dict(key="fc_units", value=fc_units)

        # ------------------
        # Define ROCKET kernels
        # ------------------
        if rocket_implementation == 1:
            self._rocket_kernels = RocketConv(num_kernels=num_kernels, max_receptive_field=max_receptive_field)
        elif rocket_implementation == 2:
            self._rocket_kernels = RocketConv2d(in_channels=1, out_channels=num_kernels,
                                                kernel_size=(1, max_receptive_field))
        elif rocket_implementation == 3:
            self._rocket_kernels = RocketConv1d(num_kernels=num_kernels, max_receptive_field=max_receptive_field)
        else:
            raise ValueError(f"Implementation {rocket_implementation=} was not recognised")

        self._rocket_implementation = rocket_implementation

        # --------------------------------------------
        # Define FC modules. Note that the FC modules will share input (rocket features).
        # This solution is less time and memory consuming, but the opposite case may be optional
        # in the future
        # --------------------------------------------
        self._fc_modules = nn.ModuleList(
            [FCBlock(block_id=fc_block_id, in_features=num_kernels * 2,
                     units=fc_units) for _ in range(num_region_representations)]
        )

    def precompute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform random convolutions and compute ROCKET-like features
        Examples:
            >>> my_model = UnivariateRocket(max_receptive_field=51, num_kernels=101, fc_units=(50,),
            ...                             fc_block_id=1)
            >>> my_model.precompute(x=torch.rand(size=(10, 11, 300))).size()
            torch.Size([10, 11, 202])
        """
        # Input check
        if x.dim() not in (3, 4):
            raise ValueError(f"Expected input tensor to have 3 or 4 dimension, but received {x.dim()}")

        # Maybe add dimensions
        if x.dim() == 3:
            x = torch.unsqueeze(x, dim=1)  # shape=(batch, 1, channels, time_steps)

        # ---------------------------
        # Perform convolution
        # ---------------------------
        x = self._rocket_kernels(x)

        if self._rocket_implementation == 2:
            # ---------------------------
            # Compute rocket-like features
            # ---------------------------
            x = compute_ppv_and_max(x)

        return x

    def forward(self, x: torch.Tensor, precomputed: torch.Tensor) -> torch.Tensor:
        """
        Forward method to calculate and apply attention vector.
        Args:
            x: The EEG data, with shape=(batch, num_channels_in_region, time_steps). Remember to only pass the allowed
                channels of the EEG data (use e.g. x[:, allowed_node_indices] as input)
            precomputed: Rocket-like features, with shape=(batch, num_channels_in_region, 2*num_kernels)
        Returns:
            Univariate time series, a torch tensor with shape=(batch, num_region_representations, time_steps)
        Examples:
            >>> my_model = UnivariateRocket(num_kernels=101, max_receptive_field=51, fc_units=(50,),
            ...                             fc_block_id=1, num_region_representations=33)
            >>> my_model(x=torch.rand(10, 11, 300), precomputed=torch.rand(10, 11, 202)).size()
            torch.Size([10, 33, 300])
            >>> # Raises RuntimeError if wrong feature dims
            >>> _ = my_model(x=torch.rand(10, 11, 300), precomputed=torch.rand(10, 11, 203))
            Traceback (most recent call last):
            ...
            RuntimeError: mat1 and mat2 shapes cannot be multiplied (110x203 and 202x50)
        """
        # Initialise list which will contain the region representations
        region_representations: List[torch.Tensor] = []

        # Loop through all FC modules
        for fc_module in self._fc_modules:
            # --------------------------------
            # Pass through FC-module and apply
            # softmax to get attention vector
            # --------------------------------
            attention = fc_module(precomputed)
            attention = torch.squeeze(attention, dim=-1)  # remove redundant dimension
            attention = torch.softmax(attention, dim=-1)  # softmax over the nodes in the group

            # --------------------------------
            # Apply attention vector on the EEG
            # data, and append as a region representation
            # --------------------------------
            region_representations.append(torch.matmul(torch.unsqueeze(attention, dim=1), x))

        return torch.cat(region_representations, dim=1)


# --------------
# Classes for shared univariate kernels
# --------------
class UnivariateRocketKernels(PoolingBase):
    """
    Applies Random Convolutional Kernels on all time series in a univariate way (one row at a time). Rocket-like
    features are then extracted. The difference between this and UnivariateRocket, is that no FC module are used here,
    it only computes the ROCKET-like features
    """

    supports_precomputing = True
    forward_args = "x",

    def __init__(self,
                 rocket_implementation: int = 3,
                 num_kernels: int = 1000,
                 max_receptive_field: int = 500):
        """
        Initialise

        (unittests in test folder)
        Args:
            rocket_implementation: Implementation of ROCKET feature extractor to use. They should be mathematically ,
                equivalent, but have different pros/cons with respect to memory usage and efficiency
            num_kernels: Number of random convolutional kernels used
            max_receptive_field: Max receptive field (upper bound) of the kernels
        """
        super().__init__()

        # ------------------
        # Define ROCKET kernels
        # ------------------
        if rocket_implementation == 1:
            self._rocket_kernels = RocketConv(num_kernels=num_kernels, max_receptive_field=max_receptive_field)
        elif rocket_implementation == 2:
            self._rocket_kernels = RocketConv2d(in_channels=1, out_channels=num_kernels,
                                                kernel_size=(1, max_receptive_field))
        elif rocket_implementation == 3:
            self._rocket_kernels = RocketConv1d(num_kernels=num_kernels, max_receptive_field=max_receptive_field)
        else:
            raise ValueError(f"Implementation {rocket_implementation=} was not recognised")

        self._rocket_implementation = rocket_implementation

    def precompute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform random convolutions and compute ROCKET-like features
        Examples:
            >>> my_model = UnivariateRocketKernels(max_receptive_field=51, num_kernels=101)
            >>> my_model.precompute(x=torch.rand(size=(10, 11, 300))).size()
            torch.Size([10, 11, 202])
        """
        # Input check
        if x.dim() not in (3, 4):
            raise ValueError(f"Expected input tensor to have 3 or 4 dimension, but received {x.dim()}")

        # Maybe add dimensions
        if x.dim() == 3:
            x = torch.unsqueeze(x, dim=1)  # shape=(batch, 1, channels, time_steps)

        # ---------------------------
        # Perform convolution
        # ---------------------------
        x = self._rocket_kernels(x)

        if self._rocket_implementation == 2:
            # ---------------------------
            # Compute rocket-like features
            # ---------------------------
            x = compute_ppv_and_max(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to calculate and apply attention vector.
        Args:
            x: The EEG data, with shape=(batch, num_channels_in_region, time_steps). Remember to only pass the allowed
                channels of the EEG data (use e.g. x[:, allowed_node_indices] as input)
        Returns:
            Univariate time series, a torch tensor with shape=(batch, num_region_representations, time_steps)
        Examples:
            >>> my_model = UnivariateRocketKernels(num_kernels=101, max_receptive_field=51)
            >>> my_model(x=torch.rand(10, 11, 300)).size()
            torch.Size([10, 11, 202])
        """
        return self.precompute(x)


class FCAttentionModule(GroupPoolingBase):
    """
    This module will not do any precomputing itself, as it expects a separate module to it instead. This is such that
    the ROCKET kernels are shared across all regions and channel splits
    """

    supports_precomputing = True
    forward_args = "precomputed", "x", "ch_names", "channel_name_to_index"

    def __init__(self,
                 num_pooling_modules: Optional[int] = None,
                 hyperparameters: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None):
        """
        Examples:
             >>> FCAttentionModule(num_pooling_modules=3, # doctest: +NORMALIZE_WHITESPACE
             ...                   hyperparameters={"in_features": 9, "fc_units": [6, 2]}).hyperparameters
             {'num_pooling_modules': 3, 'in_features': 9, 'fc_units': [6, 2], 'fc_block_id': 3}
        """
        super().__init__()

        # Get hyperparameters as a tuple of dictionaries modules
        hyperparameters = dict() if hyperparameters is None else hyperparameters
        if "fc_block_id" not in hyperparameters:
            hyperparameters["fc_block_id"] = 3
        if "fc_units" not in hyperparameters:
            hyperparameters["fc_units"] = [1]
        if "num_pooling_modules" in hyperparameters:
            del hyperparameters["num_pooling_modules"]

        # todo: not well-written and intuitive code...
        self.update_input_dict("num_pooling_modules", num_pooling_modules)
        for key, value in hyperparameters.items():
            self.update_input_dict(key, value)
        self.delete_input(key="hyperparameters")

        hyperparameters = tuple(hyperparameters for _ in range(num_pooling_modules)) \
            if isinstance(hyperparameters, dict) else hyperparameters

        # Define FC modules which will compute attention coefficients from ROCKET-like features
        self._fc_modules = nn.ModuleList(
                    [FCBlock(**params) for params in hyperparameters]
        )

    def forward(self, x: torch.Tensor, precomputed: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...],
                channel_name_to_index: Dict[str, int]):
        """
        Forward method
        Args:
            x: Full EEG data
            precomputed: Precomputed features of full EEG data
            ch_names: Channel names of regions
            channel_name_to_index: Dict mapping channel name to index

        Returns:
        Examples:
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 300))
            >>> my_precomputed = torch.rand(size=(10, 13, 202))
            >>> my_model = FCAttentionModule(num_pooling_modules=6,
            ...                              hyperparameters={"in_features": 202, "fc_units": (2, 1, 5, 87)})
            >>> my_model(my_x, my_precomputed, my_cs_names, my_channel_name_to_index).size()
            torch.Size([10, 522, 300])
            >>> my_model = FCAttentionModule(num_pooling_modules=6,
            ...                              hyperparameters={"in_features": 202, "fc_units": (2, 1, 5, 87),
            ...                                               "fc_block_id": 1})
            >>> my_model(my_x, my_precomputed, my_cs_names, my_channel_name_to_index).size()
            torch.Size([10, 6, 300])
        """
        # Loop through all regions in the path
        region_representations: List[torch.Tensor] = []
        for legal_ch_names, fc_module in zip(ch_names, self._fc_modules):
            # Extract indices of the nodes in the group
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # ---------------------
            # Compute coefficients
            # ---------------------
            # Pass through FC module
            coefficients = fc_module(precomputed[:, allowed_node_indices])

            # Normalise
            coefficients = torch.softmax(coefficients, dim=1)

            # --------------------------------
            # Apply attention vector on the EEG
            # data, and append as a region representation
            # --------------------------------
            # Add it to the slots
            region_representations.append(torch.matmul(torch.transpose(coefficients, dim0=1, dim1=2),
                                                       x[:, allowed_node_indices]))

        return torch.cat(region_representations, dim=1)

    # @property
    # def hyperparameters(self) -> Tuple[Dict[str, Any], ...]:
    #     return tuple(fc_module.hyperparameters for fc_module in self._fc_modules)


class UnivariateRocketGroup(GroupPoolingBase):

    supports_precomputing = True
    forward_args = "precomputed", "x", "ch_names", "channel_name_to_index"

    def __init__(self, num_pooling_modules: Optional[int] = None,
                 hyperparameters: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None):
        """
        Initialise
        Args:
            num_pooling_modules:
            hyperparameters:
        Examples:
            >>> my_hparams = ({"num_kernels": 4, "fc_block_id": 2, "rocket_implementation": 3,
            ...                "num_region_representations": 35},
            ...               {"max_receptive_field": 51, "fc_units": (3, 1, 5)})
            >>> my_model = UnivariateRocketGroup(hyperparameters=my_hparams)
            >>> my_model.hyperparameters  # doctest: +NORMALIZE_WHITESPACE
            ({'num_kernels': 4, 'fc_block_id': 2, 'rocket_implementation': 3, 'num_region_representations': 35,
              'max_receptive_field': 500, 'fc_units': [2]},
             {'max_receptive_field': 51, 'fc_units': [3, 1, 5], 'rocket_implementation': 3, 'num_kernels': 1000,
               'fc_block_id': 1, 'num_region_representations': 1})
            >>> my_model = UnivariateRocketGroup(num_pooling_modules=3, hyperparameters={"num_kernels": 100,
            ...                                                                          "max_receptive_field": 35})
            >>> my_model.hyperparameters  # doctest: +NORMALIZE_WHITESPACE
            ({'num_kernels': 100, 'max_receptive_field': 35, 'rocket_implementation': 3, 'fc_block_id': 1,
              'fc_units': [50], 'num_region_representations': 1},
             {'num_kernels': 100, 'max_receptive_field': 35, 'rocket_implementation': 3, 'fc_block_id': 1,
              'fc_units': [50], 'num_region_representations': 1},
             {'num_kernels': 100, 'max_receptive_field': 35, 'rocket_implementation': 3, 'fc_block_id': 1,
              'fc_units': [50], 'num_region_representations': 1})
            >>> len(my_model._pooling_modules)
            3
        """
        super().__init__()

        # Get hyperparameters as a tuple of dictionaries modules
        hyperparameters = dict() if hyperparameters is None else hyperparameters
        hyperparameters = tuple(hyperparameters for _ in range(num_pooling_modules)) \
            if isinstance(hyperparameters, dict) else hyperparameters

        # Set pooling modules
        self._pooling_modules = nn.ModuleList([
            UnivariateRocket(**params) for params in hyperparameters
        ])

    # ------------------------------------
    # Forward methods and pre-computing
    # ------------------------------------
    def precompute(self, x: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...],
                   channel_name_to_index: Dict[str, int]) -> Tuple[torch.Tensor, ...]:
        """
        Method for precomputing
        Args:
            x: EEG containing all channels
            ch_names: A tuple of legal channel names. The first element is the legal channel names of first region, the
                second element is legal channels of the second region, and so on.
            channel_name_to_index: Dictionary which maps from channel name to index

        Returns: Output of all regions
        Examples:
            >>> hyperparams = ({"num_kernels": 101}, {"num_kernels": 43, "max_receptive_field": 67},
            ...                {"num_kernels": 203}, {"num_kernels": 20, "num_region_representations": 33},
            ...                {"num_kernels": 34}, {"num_kernels": 3})
            >>> my_model = UnivariateRocketGroup(hyperparameters=hyperparams)
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 300))
            >>> my_outputs = my_model.precompute(my_x, ch_names=my_cs_names,
            ...                                  channel_name_to_index=my_channel_name_to_index)
            >>> len(my_outputs)
            6
            >>> tuple(outputs.size() for outputs in my_outputs)  # type: ignore[attr-defined]
            ... # doctest: +NORMALIZE_WHITESPACE
            (torch.Size([10, 3, 202]), torch.Size([10, 2, 86]), torch.Size([10, 4, 406]), torch.Size([10, 2, 40]),
             torch.Size([10, 3, 68]), torch.Size([10, 1, 6]))
        """
        # Input check
        if len(ch_names) != len(self._pooling_modules):
            raise ValueError("Expected number of channel name tuples to be the same as the number of pooling modules, "
                             f"but found {len(ch_names)} and {len(self._pooling_modules)}")

        # Loop through all regions, and precompute
        latent_features: List[torch.Tensor] = list()
        for pooling_module, legal_ch_names in zip(self._pooling_modules, ch_names):
            # Extract the indices of the legal channels for this region
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Run the pooling module on only the channels in the region
            latent_features.append(pooling_module.precompute(x=x[:, allowed_node_indices]))

        # Return outputs of all regions as a tuple
        return tuple(latent_features)

    def forward(self, x: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...], channel_name_to_index: Dict[str, int],
                precomputed: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Forward method
        Args:
            x: Full EEG data
            ch_names:
            channel_name_to_index:
            precomputed:

        Returns:
        Examples:
            >>> hyperparams = ({"num_kernels": 101}, {"num_kernels": 43, "max_receptive_field": 51},
            ...                {"num_kernels": 203}, {"num_kernels": 20, "num_region_representations": 33},
            ...                {"num_kernels": 34}, {"num_kernels": 3})
            >>> my_model = UnivariateRocketGroup(hyperparameters=hyperparams)
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 300))
            >>> my_features = my_model.precompute(my_x, ch_names=my_cs_names,
            ...                                   channel_name_to_index=my_channel_name_to_index)
            >>> my_model(x=my_x, ch_names=my_cs_names, channel_name_to_index=my_channel_name_to_index,
            ...          precomputed=my_features).size()  # 5 have 1 region representation, the last has 33
            torch.Size([10, 38, 300])
        """
        # Loop through all regions in the path
        region_representations: List[torch.Tensor] = []
        for legal_ch_names, pre_comp, pooling_module in zip(ch_names, precomputed, self._pooling_modules):
            # Extract indices of the nodes in the group
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Run forward pass of pooling modules
            slot = pooling_module(x=x[:, allowed_node_indices], precomputed=pre_comp)

            # Add it to the slots
            region_representations.append(slot)

        return torch.cat(region_representations, dim=1)

    # ------------
    # Properties
    # ------------
    @property
    def hyperparameters(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(pooling_module.hyperparameters for pooling_module in self._pooling_modules)


# -------------------------------------------------
# Functions
# -------------------------------------------------
def _stack_to_univariate(x: numpy.ndarray) -> numpy.ndarray:
    """
    Given a ndarray of shape=(num_subjects, channels, time_steps), this function stacks the channels into the subject
    dimension to obtain a ndarray with shape=(num_subjects*num_channels, 1, num_time_steps)
    Args:
        x: EEG data with shape=(num_subjects, channels, num_time_steps)

    Returns: Stacked/reshaped version of the input
    Examples:
        >>> my_x = numpy.concatenate([numpy.ones(shape=(1, 4, 10))*i for i in range(5)])  # type: ignore[attr-defined]
        >>> _stack_to_univariate(my_x)
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
               [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
               [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
               [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
               [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]])
    """
    # Get the shape
    subjects, channels, time_steps = x.shape

    # Reshape and return
    return numpy.reshape(x, newshape=(subjects * channels, time_steps))


def _unstack_to_multivariate(x: numpy.ndarray, num_channels: int) -> numpy.ndarray:
    """
    The inverse of _stack_to_univariate()
    Args:
        x:
        num_channels:

    Returns:
    Examples:
        >>> my_inputs = [numpy.ones(shape=(1, 4, 10))*i for i in range(5)]  # type: ignore[attr-defined]
        >>> my_x = _stack_to_univariate(numpy.concatenate(my_inputs))
        >>> _unstack_to_multivariate(my_x, num_channels=4)
        array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        <BLANKLINE>
               [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
        <BLANKLINE>
               [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]],
        <BLANKLINE>
               [[3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]],
        <BLANKLINE>
               [[4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]]])
    """
    return numpy.reshape(x, newshape=(x.shape[0] // num_channels, num_channels, x.shape[-1]))
