"""
Implementing common FC blocks
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import InputStore


class _FCBlock1(nn.Module, InputStore):
    """
    FC Layer using relu as activation function, except from the final layer, which has linear activation function. No
    dropout used
    """

    def __init__(self, in_features: int, units: List[int], bias: bool = True):
        super().__init__()

        units = list(units)

        # --------------------------------------
        # Define FC layers
        # --------------------------------------
        units = units + [1] if units[-1] != 1 else units
        in_features = [in_features] + units[:-1].copy()

        self._layers = nn.ModuleList([nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
                                      for in_feature, out_feature in zip(in_features, units)])
        self._num_layers = len(self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FCBlock forward method
        Args:
            x: tensor with shape=(batch, d) where d is input feature dimension. Input shape can also be
                shape=(batch, group_size, d)
        Returns: tensor with shape=(batch, 1) or shape=(batch, group_size, 1)
        Examples:
            >>> my_batch, my_d = 10, 3
            >>> my_data = torch.rand(size=(my_batch, my_d))
            >>> my_model = _FCBlock1(in_features=my_d, units=[4, 5, 2, 7])
            >>> my_model(my_data).size()
            torch.Size([10, 1])
            >>> my_batch, my_channels, my_d = 10, 5, 3
            >>> my_data = torch.rand(size=(my_batch, my_channels, my_d))
            >>> my_model = _FCBlock1(in_features=my_d, units=[4, 5, 2, 7], bias=False)
            >>> my_model(my_data).size()
            torch.Size([10, 5, 1])
        """
        for i, layer in enumerate(self._layers):
            # Passing through layer
            x = layer(x)

            # If the current layer is not the final layer, use relu
            if i != self._num_layers - 1:
                x = F.relu(x)

        return x


class _FCBlock2(nn.Module, InputStore):
    """
    FC Layer using relu as activation function, except from the final layer, which has linear activation function.
    Dropout used after all layer except the final one is applied with dropout=0.5 as default
    """

    def __init__(self, in_features: int, units: List[int], bias: bool = True, dropout: float = 0.5):
        super().__init__()

        units = list(units)
        self._dropout = dropout

        # --------------------------------------
        # Define FC layers
        # --------------------------------------
        units = units + [1] if units[-1] != 1 else units
        in_features = [in_features] + units[:-1].copy()

        self._layers = nn.ModuleList([nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
                                      for in_feature, out_feature in zip(in_features, units)])
        self._num_layers = len(self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FCBlock forward method
        Args:
            x: tensor with shape=(batch, d) where d is input feature dimension. Input shape can also be
                shape=(batch, group_size, d)
        Returns: tensor with shape=(batch, 1) or shape=(batch, group_size, 1)
        Examples:
            >>> my_batch, my_d = 10, 3
            >>> my_data = torch.rand(size=(my_batch, my_d))
            >>> my_model = _FCBlock2(in_features=my_d, units=[4, 5, 2, 7])
            >>> my_model(my_data).size()
            torch.Size([10, 1])
            >>> my_batch, my_channels, my_d = 10, 5, 3
            >>> my_data = torch.rand(size=(my_batch, my_channels, my_d))
            >>> my_model = _FCBlock2(in_features=my_d, units=[4, 5, 2, 7], bias=False)
            >>> my_model(my_data).size()
            torch.Size([10, 5, 1])
        """
        for i, layer in enumerate(self._layers):
            # Passing through layer
            x = layer(x)

            # If the current layer is not the final layer, use relu and dropout
            if i != self._num_layers - 1:
                x = F.relu(x)
                x = F.dropout(input=x, p=self._dropout)

        return x


class _FCBlock3(nn.Module, InputStore):
    """
    FC Layer using relu as activation function, except from the final layer, which has linear activation function. No
    dropout used. The output is NOT forced to be scalar
    """

    def __init__(self, in_features: int, units: Optional[List[int]] = None, bias: bool = True):
        super().__init__()

        # --------------------------------------
        # Define FC layers
        # --------------------------------------
        units = [1] if units is None else units
        units = list(units)
        in_features = [in_features] + units[:-1].copy()

        self._layers = nn.ModuleList([nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
                                      for in_feature, out_feature in zip(in_features, units)])
        self._num_layers = len(self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FCBlock forward method
        :param x: tensor with shape=(batch, d) where d is input feature dimension. Input shape can also be
            shape=(batch, group_size, d)
        :return: tensor with shape=(batch, 1) or shape=(batch, group_size, 1)
        Examples:
            >>> my_batch, my_d = 10, 3
            >>> my_data = torch.rand(size=(my_batch, my_d))
            >>> my_model = _FCBlock3(in_features=my_d, units=[4, 5, 2, 7])
            >>> my_model(my_data).size()
            torch.Size([10, 7])
            >>> my_batch, my_channels, my_d = 10, 5, 3
            >>> my_data = torch.rand(size=(my_batch, my_channels, my_d))
            >>> my_model = _FCBlock3(in_features=my_d, units=[4, 5, 2, 77], bias=False)
            >>> my_model(my_data).size()
            torch.Size([10, 5, 77])
        """
        for i, layer in enumerate(self._layers):
            # Passing through layer
            x = layer(x)

            # If the current layer is not the final layer, use relu
            if i != self._num_layers - 1:
                x = F.relu(x)

        return x


class FCBlock(nn.Module):

    def __init__(self, block_id: int = 3, **kwargs):
        """
        initialise
        Args:
            block_id: Which of the implemented FC-blocks to use
            kwargs: Arguments which should be passed to the FC-block, such as in_features and units
        Examples:
            >>> my_model = FCBlock(block_id=1, in_features=5, units=[4, 2, 6])
            >>> my_model.hyperparameters
            {'in_features': 5, 'units': [4, 2, 6], 'bias': True, 'block_id': 1}
            >>> type(my_model._fc_block)
            <class 'fc_blocks._FCBlock1'>
            >>> my_model = FCBlock(block_id=3, in_features=5, units=[4, 2, 6])
            >>> my_model.hyperparameters
            {'in_features': 5, 'units': [4, 2, 6], 'bias': True, 'block_id': 3}
            >>> type(my_model._fc_block)
            <class 'fc_blocks._FCBlock3'>
            >>> _ = FCBlock(block_id=-1, in_features=9, units=[1, 11, 111, 5])
            Traceback (most recent call last):
            ...
            ValueError: The block id -1 was not recognised
        """
        super().__init__()

        # Maybe change some of the keys
        if "fc_units" in kwargs and "units" not in kwargs:
            kwargs["units"] = kwargs["fc_units"]
            del kwargs["fc_units"]
        if "fc_block_id" in kwargs:
            block_id = kwargs["fc_block_id"]
            del kwargs["fc_block_id"]

        # --------------------------------------
        # Get the correct FC Block
        # --------------------------------------
        self._fc_block = self._get_fc_block(block_id=block_id, **kwargs)
        self._block_id = block_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(Tested separately for each class)"""
        return self._fc_block(x)

    @staticmethod
    def _get_fc_block(block_id: int, **kwargs) -> nn.Module:
        if block_id == 1:
            return _FCBlock1(**kwargs)
        elif block_id == 2:
            return _FCBlock2(**kwargs)
        elif block_id == 3:
            return _FCBlock3(**kwargs)
        else:
            raise ValueError(f"The block id {block_id} was not recognised")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        h_params = self._fc_block.inputs.copy()
        h_params["block_id"] = self._block_id
        return h_params
