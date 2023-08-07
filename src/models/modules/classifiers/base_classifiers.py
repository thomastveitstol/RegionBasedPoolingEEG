import abc
from typing import Any, Dict

import torch.nn as nn


class BaseClassifier(abc.ABC, nn.Module):

    activation_function: str

    def __init__(self, **kwargs):
        super().__init__()

        # Store hyperparameters
        self._hyperparameters = kwargs

    # -------------
    # Properties
    # -------------
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._hyperparameters


class BaseMTSClassifier(BaseClassifier):
    ...


class BaseVectorClassifier(BaseClassifier):
    ...
