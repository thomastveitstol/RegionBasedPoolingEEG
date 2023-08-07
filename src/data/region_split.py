import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from matplotlib import pyplot

from src.data.datasets.data_base import BaseChannelSystem
from src.data.line_separation.centroid_polygons import CentroidPolygons, CentroidPolygonsConf2, NumericalError
from src.data.line_separation.utils import project_head_shape
from src.utils import CartesianCoordinates, PolarCoordinates, ChGroup, to_cartesian


# --------------------------------
# Main class for all topological splits
# --------------------------------
class TopologicalRegions:
    """
    Main Class for topological regions
    """

    __slots__ = "_regions"

    class_candidates = (CentroidPolygons, CentroidPolygonsConf2)

    def __init__(self,
                 split_mode: str,
                 nodes: Dict[str, Tuple[float, float]],
                 min_nodes: int,
                 **kwargs) -> None:
        """
        Initialise
        Args:
            num_regions: Number of regions
            split_mode: Selection of split-type
            **kwargs: Depends on the split_mode. See the individual region splits
        Examples:
            >>> import mne
            >>> from src.data.line_separation.utils import project_head_shape
            >>> import random
            >>> random.seed(3)
            >>> numpy.random.seed(3)
            >>> my_nodes_3d = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
            >>> my_nodes = project_head_shape(electrode_positions=my_nodes_3d)
            >>> _ = TopologicalRegions(split_mode="CentroidPolygons", nodes=my_nodes, min_nodes=2, k=(5,))
            >>> # The number of available candidates is small, per now...
            >>> my_regions = TopologicalRegions(split_mode="NodeSplitFixedAngle", nodes=my_nodes, min_nodes=10)
            Traceback (most recent call last):
            ...
            KeyError: 'Split mode NodeSplitFixedAngle was not recognised'
            >>> my_regions = TopologicalRegions(split_mode="NodeSplitFixedPoint", nodes=my_nodes, min_nodes=10)
            Traceback (most recent call last):
            ...
            KeyError: 'Split mode NodeSplitFixedPoint was not recognised'
            >>> _ = TopologicalRegions(split_mode="NotASplitMode", nodes=my_nodes, min_nodes=4)
            Traceback (most recent call last):
            ...
            KeyError: 'Split mode NotASplitMode was not recognised'
        """

        for region_split_class in self.class_candidates:
            if region_split_class.__name__ == split_mode:
                self._regions = region_split_class(nodes=nodes, min_nodes=min_nodes, **kwargs)
                break
        else:
            raise KeyError(f"Split mode {split_mode} was not recognised")

    def place(self, coordinates: Union[CartesianCoordinates, PolarCoordinates]) -> ChGroup:
        """Given the cartesian coordinates of a single electrode, return its correct ChGroup"""
        return self._regions.coord_to_ch_group(coordinates)

    @classmethod
    def get_candidate_settings(cls, split_mode: str) -> Tuple[Dict[str, Any], ...]:
        """Method for getting a selection of pre-defined kwargs"""
        for region_split_class in cls.class_candidates:
            if region_split_class.__name__ == split_mode:
                return region_split_class.candidate_settings

        raise KeyError(f"Split mode {split_mode} was not recognised")


# --------------------------------
# Main class for channel splitting
# --------------------------------
class ChannelSplit:
    """
    Channel Split class

    This class should split channel systems based on the topological regions/constraints which are to be imposed. It is
    possible to e.g. split a channel system based on both azimuthal angle and z-coordinate.
    """

    __slots__ = "_topological_regions", "_channel_splits", "_allowed_channel_groups"

    def __init__(self,
                 nodes: Dict[str, Tuple[float, float]],
                 min_nodes: int,
                 candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
                 **kwargs) -> None:
        """
        Initialise
        Args:
            nodes: Dictionary of electrode positions. The keys are channel names, the values are 2D positions
            min_nodes: The minimum number of nodes allowed in a region
            candidate_region_splits: IDs of the candidate regions splits. If None, all available will be selected. If
                int, it is interpreted as there is only one candidate
            **kwargs: Keyword arguments passed to TopologicalRegions. If none is passed, a predefined selection will be
                made

        Examples:
            >>> import mne
            >>> my_nodes_3d = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
            >>> my_nodes_2d = project_head_shape(electrode_positions=my_nodes_3d)
            >>> my_points_ = numpy.array(tuple(my_nodes_2d.values()))
            >>> my_nodes = {f'Ch_{i_}': point for i_, point in enumerate(my_points_)}  # type: ignore[attr-defined]
            >>> _ = ChannelSplit(nodes=my_nodes, min_nodes=14, candidate_region_splits="CentroidPolygons")
            >>> _ = ChannelSplit(nodes=my_nodes, min_nodes=14, k=(3, 3, 3, 3),
            ...                  candidate_region_splits="CentroidPolygons")
            >>> _ = ChannelSplit(nodes=my_nodes, min_nodes=14, k=(3, 3, 3, 3), candidate_region_splits=None)
        """
        # Handle the non-tuple cases
        candidate_region_splits = (candidate.__name__ for candidate in TopologicalRegions.class_candidates) \
            if candidate_region_splits is None else candidate_region_splits
        candidate_region_splits = (candidate_region_splits,) if isinstance(candidate_region_splits, str) \
            else candidate_region_splits

        # -------------
        # Propose candidates
        # -------------
        if kwargs:
            candidates = tuple(
                TopologicalRegions(split_mode=region_split, nodes=nodes, min_nodes=min_nodes, **kwargs)
                for region_split in candidate_region_splits
            )
        else:
            candidates = tuple(
                TopologicalRegions(split_mode=region_split, nodes=nodes, min_nodes=min_nodes, **settings)
                for region_split in candidate_region_splits
                for settings in TopologicalRegions.get_candidate_settings(region_split)
            )

        # Shuffle the candidates
        candidates = list(candidates)
        random.shuffle(candidates)
        candidates = tuple(candidates)

        # -------------
        # Select candidate at random
        # -------------
        self._topological_regions: TopologicalRegions = random.choice(candidates)

        # -------------
        # Initialise
        # -------------
        self._allowed_channel_groups = None
        self._channel_splits = dict()

    @classmethod
    def generate_multiple_channel_splits(cls,
                                         num_channel_splits: int,
                                         nodes: Dict[str, Tuple[float, float]],
                                         min_nodes: int = 1,
                                         candidate_region_splits: Optional[Union[str, Tuple[str, ...]]] = None,
                                         **kwargs):
        """
        Method for generating multiple channel splits
        Examples:
            >>> import mne
            >>> my_nodes_3d = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
            >>> my_nodes_2d = project_head_shape(electrode_positions=my_nodes_3d)
            >>> my_points_ = numpy.array(tuple(my_nodes_2d.values()))
            >>> my_nodes = {f'Ch_{i_}': point for i_, point in enumerate(my_points_)}  # type: ignore[attr-defined]
            >>> _ = ChannelSplit.generate_multiple_channel_splits(num_channel_splits=3, nodes=my_nodes)
        """
        channel_splits = list()
        for i in range(num_channel_splits):
            # Get the next channel split. Sometimes, the generation of a channel split struggles with numerical errors.
            # If it happens, we just try again.
            while True:
                try:
                    next_channel_split = cls(nodes=nodes, min_nodes=min_nodes,
                                             candidate_region_splits=candidate_region_splits, **kwargs)
                    break
                except NumericalError:
                    pass

            channel_splits.append(next_channel_split)

        return tuple(channel_splits)

    # -----------------------------
    # Methods for fitting channel systems
    # -----------------------------
    def _get_allowed_groups(self, electrode_positions: Dict[str, Union[PolarCoordinates, CartesianCoordinates]]) \
            -> Tuple[ChGroup, ...]:
        allowed_groups = []
        # Loop through all electrodes in the current channel system
        for position in electrode_positions.values():
            # Place the channel in a tuple group
            group = self._topological_regions.place(coordinates=position)

            # If the channel group has not been seen before, add it to allowed groups
            if group not in allowed_groups:
                allowed_groups.append(group)
        return tuple(allowed_groups)

    def fit_allowed_groups(self, channel_systems: Union[BaseChannelSystem,
                                                        List[BaseChannelSystem],
                                                        Tuple[BaseChannelSystem, ...]]) -> None:
        """
        Fit channel systems. When using this method, all regions which do not contain a single electrode in all channel
        systems will be removed.
        Args:
            channel_systems: A list of objects which inherits from BaseChannelSystem. All groups which do not contain an
                electrode in these channel systems will be removed

        Returns: Nothing, it just removes groups not seen in the given channel systems
        Examples:
            >>> import mne
            >>> from src.data.datasets.example_data import ExampleChannelSystem
            >>> from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem
            >>> numpy.random.seed(2)
            >>> my_nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
            >>> my_nodes_2d = project_head_shape(my_nodes_3d)
            >>> my_nodes_3d = {key: value.coordinates for key, value in my_nodes_3d.items()}  # type: ignore
            >>> my_points_ = numpy.array(tuple(my_nodes_2d.values()))
            >>> my_nodes = {f'Ch_{i_}': point for i_, point in enumerate(my_points_)}  # type: ignore[attr-defined]
            >>> my_channel_split = ChannelSplit(nodes=my_nodes, min_nodes=1, k=(10, 10),
            ...                                 candidate_region_splits="CentroidPolygons")
            >>> my_channel_split.fit_allowed_groups(channel_systems=CleanedChildChannelSystem())
            >>> len(my_channel_split._allowed_channel_groups)  # All groups are OK
            100
            >>> my_channel_split.fit_allowed_groups(channel_systems=ExampleChannelSystem())
            >>> len(my_channel_split._allowed_channel_groups)  # Not all groups contain an electrode
            54
            >>> # When running again, the removal from previous calls are forgotten
            >>> my_channel_split.fit_allowed_groups(channel_systems=CleanedChildChannelSystem())
            >>> len(my_channel_split._allowed_channel_groups)  # All groups are OK
            100
            >>> # Can also pass in several channel systems
            >>> my_channel_systems = (CleanedChildChannelSystem(), ExampleChannelSystem())
            >>> my_channel_split.fit_allowed_groups(channel_systems=my_channel_systems)
            >>> len(my_channel_split._allowed_channel_groups)
            54
        """
        if isinstance(channel_systems, BaseChannelSystem):
            channel_systems = (channel_systems,)

        # Initialise the list of allowed channel groups
        allowed_groups = {}

        # Loop through all channel systems and obtain the allowed ChGroups
        for channel_system in channel_systems:
            channel_positions = channel_system.get_electrode_positions()
            allowed_groups[channel_system.name] = self._get_allowed_groups(electrode_positions=channel_positions)

        # Only those groups which are seen in all channel systems should be included
        all_groups = set.intersection(*map(set, allowed_groups.values()))

        self._allowed_channel_groups = tuple(all_groups)

    def fit_channel_system(self, channel_system: BaseChannelSystem) -> None:
        """
        Fit a channel system on the channel splits
        Args:
            channel_system: A Channel System object, which inherits from BaseChannelSystem

        Returns:
            Nothing
        Examples:
            >>> import mne
            >>> from src.data.datasets.example_data import ExampleChannelSystem
            >>> numpy.random.seed(2)
            >>> my_nodes_3d = ExampleChannelSystem().get_electrode_positions()
            >>> my_nodes_2d = project_head_shape(my_nodes_3d)
            >>> my_nodes_3d = {key: value.coordinates for key, value in my_nodes_3d.items()}  # type: ignore
            >>> my_points_ = numpy.array(tuple(my_nodes_2d.values()))
            >>> my_nodes = {f'Ch_{i_}': point for i_, point in enumerate(my_points_)}  # type: ignore[attr-defined]
            >>> my_channel_split = ChannelSplit(nodes=my_nodes, min_nodes=3, k=(10, 5),
            ...                                 candidate_region_splits="CentroidPolygons")
            >>> my_channel_split.fit_allowed_groups(channel_systems=ExampleChannelSystem())
            >>> my_channel_split.fit_channel_system(ExampleChannelSystem())
            >>> len(my_channel_split.allowed_ch_groups)
            50
            >>> tuple(my_channel_split.channel_splits.keys())
            ('Example',)
            >>> tuple(my_channel_split.channel_splits["Example"].keys())[:6]  # doctest: +NORMALIZE_WHITESPACE
            (ChGroup(group_number=2304), ChGroup(group_number=6912), ChGroup(group_number=36),
             ChGroup(group_number=13824), ChGroup(group_number=6), ChGroup(group_number=108))
            >>> my_channel_split.channel_splits["Example"][ChGroup(2)]  # doctest: +NORMALIZE_WHITESPACE
            ['Ch26', 'Ch123', 'Ch169', 'Ch189']
        """
        # Initialise the values of the input channel system (set it as empty Dict)
        self._channel_splits[channel_system.name]: Dict[ChGroup, List[str]] = {}  # Keys are ch group, values ch names
        channel_split = self._channel_splits[channel_system.name]  # Dicts are mutable, so a change to LHS is sufficient

        electrode_positions = channel_system.get_electrode_positions()

        # Add all channel names to their corresponding ChGroup
        for channel_name, position in electrode_positions.items():
            # Compute the tuple of regions (each topological constraint puts it in its region)
            channel_group = self._topological_regions.place(coordinates=position)

            # If the region is allowed, append the channel name to the correct region. Otherwise, do nothing
            if channel_group in self._allowed_channel_groups:
                if channel_group not in channel_split:
                    channel_split[channel_group] = list()

                channel_split[channel_group].append(channel_name)

    def fit_channel_systems(self, channel_systems: Union[BaseChannelSystem, List[BaseChannelSystem],
                                                         Tuple[BaseChannelSystem, ...]]) -> None:
        """Fit multiple channel systems"""
        if isinstance(channel_systems, BaseChannelSystem):
            self.fit_channel_system(channel_system=channel_systems)
        else:
            for channel_system in channel_systems:
                self.fit_channel_system(channel_system)

    # -----------------------------
    # Plotting
    # -----------------------------
    def plot_channel_split(self, channel_system: BaseChannelSystem, annotate: bool = False) -> None:
        """Plot how the electrodes of a channel system has been split by ChannelSplit"""
        # Get the channel split of the given channel system
        channel_split = self.channel_splits[channel_system.name]

        # Get electrode positions of all the channels
        electrode_positions = channel_system.get_electrode_positions()

        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Loop through all regions
        for region, channels in channel_split.items():
            # Extract positions and plot
            x = [to_cartesian(electrode_positions[channel]).coordinates[0] for channel in channels]
            y = [to_cartesian(electrode_positions[channel]).coordinates[1] for channel in channels]
            z = [to_cartesian(electrode_positions[channel]).coordinates[2] for channel in channels]

            ax.scatter(x, y, z, label=region)

            # Annotate the channels with channel names (if desired)
            if annotate:
                for x_pos, y_pos, z_pos, channel in zip(x, y, z, channels):
                    ax.text(x=x_pos, y=y_pos, z=z_pos, s=channel)

        # Plotting cosmetics
        pyplot.xlabel("x")
        pyplot.ylabel("y")

        pyplot.legend()
        pyplot.title(channel_system.name)

    # -----------------
    # Properties
    # -----------------
    @property
    def allowed_ch_groups(self) -> Tuple[ChGroup, ...]:
        """Get a list of legal channel groups"""
        return self._allowed_channel_groups

    @property
    def channel_splits(self) -> Dict[str, Dict[ChGroup, Tuple[str, ...]]]:
        """The channel splits. Keys are name of the fitted channel systems, the values are dictionaries mapping from
        ChGroup to a list of str (channel/electrode names, such as 'Cz') which are part of the ChGroup"""
        return self._channel_splits


# --------------------------------
# Functions
# --------------------------------
def _splits_to_partitions(splits: Tuple[TopologicalRegions, ...], nodes: Dict[str, Tuple[float, float, float]]) \
        -> Tuple[Dict[ChGroup, List[str]], ...]:
    partitions: Tuple[Dict[ChGroup, List[str]], ...] = tuple(dict() for _ in range(len(splits)))

    # Loop through all nodes
    for name, pos in nodes.items():
        # Loop through all splits
        for split, partition in zip(splits, partitions):
            # Get the regions of the current node
            group = split.place(coordinates=CartesianCoordinates(pos))

            # (Maybe) add group to dict keys
            if group not in partition:
                partition[group] = list()

            # Store
            partition[group].append(name)

    return partitions
