"""
Script for plotting channel splits of the ChannelSystems
"""
from matplotlib import pyplot
import numpy

from src.data.line_separation.utils import project_head_shape
from src.data.region_split import ChannelSplit
from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem


def main() -> None:
    # Load electrode positions
    my_nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    my_nodes_2d = project_head_shape(my_nodes_3d)
    my_nodes_3d = {key: value.coordinates for key, value in my_nodes_3d.items()}
    my_points_ = numpy.array(tuple(my_nodes_2d.values()))
    my_nodes = {f'Ch_{i}': point for i, point in enumerate(my_points_)}

    # Define channel systems
    channel_systems = (CleanedChildChannelSystem(),)

    # Define channel split
    channel_split = ChannelSplit(nodes=my_nodes, min_nodes=5, k=(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                                 candidate_region_splits="CentroidPolygons")

    # Fit allowed groups
    channel_split.fit_allowed_groups(channel_systems=channel_systems)

    # Fit the channel system on the channel split
    channel_split.fit_channel_systems(channel_systems)

    # Plot the result
    for channel_system in channel_systems:
        channel_split.plot_channel_split(channel_system, annotate=True)

    pyplot.show()


if __name__ == "__main__":
    main()
