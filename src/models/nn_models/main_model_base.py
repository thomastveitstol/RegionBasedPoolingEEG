import abc

import torch.nn as nn


class MainModelBase(abc.ABC, nn.Module):

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Method for saving the model to a given path"""

    @classmethod
    @abc.abstractmethod
    def from_disk(cls, path: str) -> 'MainModelBase':
        """Method for loading previously saved model from disk"""

