"""
Pooling mechanism is mean
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.data.datasets.data_base import channel_names_to_indices
from src.models.modules.pooling_modules.pooling_base import PoolingBase, GroupPoolingBase


class Mean(PoolingBase):

    supports_precomputing = True
    forward_args = "precomputed",

    def __init__(self, adjusted: bool = False):
        super().__init__()

        self._adjusted = adjusted

    def precompute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Precompute
        Examples:
            >>> Mean().precompute(x=torch.normal(mean=0, std=1, size=(10, 64, 400))).size()
            torch.Size([10, 1, 400])
            >>> Mean(adjusted=True).precompute(x=torch.normal(mean=0, std=1, size=(10, 64, 400))).size()
            torch.Size([10, 1, 400])
        """
        mean = torch.mean(x, dim=1, keepdim=True)

        if self._adjusted:
            # todo
            variance = torch.var(x, dim=1, keepdim=True, unbiased=True)
            p = x.size()[1]

            expression = mean * mean - 1/p * variance
            return torch.sign(expression) * torch.sqrt(torch.abs(expression))
        else:
            return mean

    @staticmethod
    def forward(precomputed: torch.Tensor) -> torch.Tensor:
        return precomputed


class MeanGroup(GroupPoolingBase):

    supports_precomputing = True
    forward_args = ("precomputed", "x", "ch_names", "channel_name_to_index")

    def __init__(self, num_pooling_modules: Optional[int] = None,
                 hyperparameters: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None):
        super().__init__()

        # Get hyperparameters as a tuple of dictionaries modules
        hyperparameters = dict() if hyperparameters is None else hyperparameters
        hyperparameters = tuple(hyperparameters for _ in range(num_pooling_modules)) \
            if isinstance(hyperparameters, dict) else hyperparameters

        # Define pooling modules
        self._pooling_modules = nn.ModuleList([
            Mean(**params) for params in hyperparameters
        ])

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
            >>> my_model = MeanGroup(num_pooling_modules=6)
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 300))
            >>> my_outputs = my_model.precompute(my_x, ch_names=my_cs_names,
            ...                                  channel_name_to_index=my_channel_name_to_index)
            >>> tuple(outputs.size() for outputs in my_outputs)  # type: ignore[attr-defined]
            ... # doctest: +NORMALIZE_WHITESPACE
            (torch.Size([10, 1, 300]), torch.Size([10, 1, 300]), torch.Size([10, 1, 300]), torch.Size([10, 1, 300]),
             torch.Size([10, 1, 300]), torch.Size([10, 1, 300]))

        """
        # Input check
        if len(ch_names) != len(self._pooling_modules):
            raise ValueError("Expected number of channel name tuples to be the same as the number of pooling modules, "
                             f"but found {len(ch_names)} and {len(self._pooling_modules)}")

        # Loop through all regions, and precompute
        region_representations: List[torch.Tensor] = list()
        for pooling_module, legal_ch_names in zip(self._pooling_modules, ch_names):
            # Extract the indices of the legal channels for this region
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Run the pooling module on only the channels in the region
            region_representations.append(pooling_module.precompute(x=x[:, allowed_node_indices]))

        # Return outputs of all regions as a tuple
        return tuple(region_representations)

    def forward(self, precomputed: Tuple[torch.Tensor, ...], x: torch.Tensor = None,
                ch_names: Tuple[Tuple[str, ...], ...] = None, channel_name_to_index: Dict[str, int] = None) \
            -> torch.Tensor:
        """
        Forward method for computing region representations for precomputed features. The region representations are
        concatenated
        Args:
            precomputed: Tuple of precomputed features (averages). The first element in the tuple is the precomputed
            features of the first region, the second element is the precomputed features of the second region and so on

        Returns: Region representations concatenated to a single tensor of shape=(batch, num_regions, time_steps)
        Examples:
            >>> batch_size = 10
            >>> hyperparams = ({"adjusted": True}, {"adjusted": True}, {"adjusted": True})
            >>> my_model = MeanGroup(hyperparameters=hyperparams)
            >>> my_features = (torch.rand((batch_size, 1, 300)), torch.rand((batch_size, 1, 300)),
            ...                torch.rand((batch_size, 1, 300)))
            >>> my_model(precomputed=my_features).size()
            torch.Size([10, 3, 300])
        """
        if precomputed:
            return torch.cat(tensors=precomputed, dim=1)

        precomputed = self.precompute(x=x, ch_names=ch_names, channel_name_to_index=channel_name_to_index)
        return torch.cat(tensors=precomputed, dim=1)

    # ------------
    # Properties
    # ------------
    @property
    def hyperparameters(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(pooling_module.hyperparameters for pooling_module in self._pooling_modules)
