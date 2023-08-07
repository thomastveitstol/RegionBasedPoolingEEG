"""
Base class for splitting into regions
"""
import abc
from typing import Any, Dict, Tuple, Union

from src.utils import CartesianCoordinates, PolarCoordinates, ChGroup


# Base class
class RegionsBase(abc.ABC):
    """
    Base class for splitting the EEG cap into regions
    """

    candidate_settings: Tuple[Dict[str, Any], ...] = tuple()

    @abc.abstractmethod
    def __init__(self, nodes: Dict[str, Tuple[float, float]], min_nodes: int):
        """
        Initialise
        Args:
            nodes: Nodes to fit the split on
            min_nodes: Minimum number of regions allowed
        """

    @abc.abstractmethod
    def coord_to_ch_group(self, coordinates: Union[CartesianCoordinates, PolarCoordinates]) -> ChGroup:
        """
        Map coordinates to channel group
        Args:
            coordinates: Instance of CartesianCoordinates (x, y, z) or PolarCoordinates (rho, theta, phi)

        Returns:
            The channel group it belongs to
        """