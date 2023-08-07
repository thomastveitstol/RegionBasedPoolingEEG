import abc
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.utils import InputStore


class PoolingBase(abc.ABC, nn.Module, InputStore):
    supports_precomputing: bool
    forward_args: Tuple[str, ...]

    @abc.abstractmethod
    def precompute(self, x: torch.Tensor) -> torch.Tensor:
        """Method for pre-computing"""

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self.inputs


class GroupPoolingBase(abc.ABC, nn.Module, InputStore):
    supports_precomputing: bool
    forward_args: Tuple[str, ...]

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self.inputs
