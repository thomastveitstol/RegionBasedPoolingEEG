"""
For using all region pooling modules
"""
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.modules.pooling_modules.connected_search import ConnectedSearch2
from src.models.modules.pooling_modules.contiuous_attention import ContinuousAttention
from src.models.modules.pooling_modules.mean import MeanGroup
from src.models.modules.pooling_modules.pooling_base import GroupPoolingBase
from src.models.modules.pooling_modules.univariate_rocket import FCAttentionModule, UnivariateRocketGroup


class GroupPoolingModule(nn.Module):

    def __init__(self, pooling_method: str, num_pooling_modules: Optional[int] = None,
                 hyperparameters: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None):
        """
        Initialise
        Args:
            pooling_method: Name of group pooling module
            hyperparameters: hyperparameters of the group pooling module
        Examples:
            >>> _ = GroupPoolingModule("Mean", num_pooling_modules=7)
            >>> _ = GroupPoolingModule("SharedRocketKernels", num_pooling_modules=5, hyperparameters={"in_features": 9})
            >>> my_hparams = {"latent_search_features": 38, "in_features": 17, "share_edge_embeddings": True,
            ...               "bias": False}
            >>> _ = GroupPoolingModule("ConnectedSearch2", num_pooling_modules=53, hyperparameters=my_hparams)
            >>> _ = GroupPoolingModule("ContinuousAttention", num_pooling_modules=53)
        """
        super().__init__()

        self._pooling_module = self._get_group_pooling_module(pooling_method=pooling_method,
                                                              num_pooling_modules=num_pooling_modules,
                                                              hyperparameters=hyperparameters)
        self._name = pooling_method

    @staticmethod
    def _get_group_pooling_module(pooling_method: str, num_pooling_modules: Optional[int],
                                  hyperparameters: Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]]) \
            -> GroupPoolingBase:
        # Select pooling module
        if pooling_method in ("Mean", "MeanGroup"):
            return MeanGroup(num_pooling_modules=num_pooling_modules, hyperparameters=hyperparameters)
        elif pooling_method in ("UnivariateRocket", "UnivariateRocketGroup"):
            return UnivariateRocketGroup(num_pooling_modules=num_pooling_modules, hyperparameters=hyperparameters)
        elif pooling_method in ("FCAttentionModule", "SharedRocketKernels"):
            return FCAttentionModule(num_pooling_modules=num_pooling_modules, hyperparameters=hyperparameters)
        elif pooling_method == "ConnectedSearch2":
            hyperparameters = dict() if hyperparameters is None else hyperparameters
            if "num_regions" in hyperparameters:
                return ConnectedSearch2(**hyperparameters)
            else:
                return ConnectedSearch2(num_regions=num_pooling_modules, **hyperparameters)
        elif pooling_method == "ContinuousAttention":
            hyperparameters = dict() if hyperparameters is None else hyperparameters
            if "num_regions" in hyperparameters:
                return ContinuousAttention(**hyperparameters)
            else:
                return ContinuousAttention(num_regions=num_pooling_modules, **hyperparameters)
        else:
            raise ValueError(f"The pooling method {pooling_method} was not recognised")

    # ---------------------
    # Forward passes
    # ---------------------
    def precompute(self, x: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...],
                   channel_name_to_index: Dict[str, int]):
        return self._pooling_module.precompute(x=x, ch_names=ch_names, channel_name_to_index=channel_name_to_index)

    def forward(self, **inputs):
        return self._pooling_module(**inputs)

    # ---------------------
    # Properties
    # ---------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def hyperparameters(self):
        return self._pooling_module.hyperparameters

    @property
    def supports_precomputing(self) -> bool:
        return self._pooling_module.supports_precomputing

    @property
    def forward_args(self) -> Tuple[str, ...]:
        return self._pooling_module.forward_args
