"""
Script for plotting an example how the regions are generated. The script with its current seed and hyperparameters
should yield the Figure as used in the paper with k=(5, 3).
"""
import random

from matplotlib import pyplot
import numpy

from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem
from src.data.line_separation.centroid_polygons import CentroidPolygons
from src.data.line_separation.utils import plot_2d_projection, project_head_shape
from src.utils import CartesianCoordinates


def main() -> None:
    # Make reproducible
    seed_ = 7
    numpy.random.seed(seed_)
    random.seed(seed_)

    # ------------------
    # Get/compute the electrode positions
    # ------------------
    # 3D
    my_nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    my_nodes_3d = {key: (value.coordinates[1], value.coordinates[0], value.coordinates[2]) for key, value in
                   my_nodes_3d.items()}

    # 2D and numpy arrays
    my_nodes_2d = project_head_shape(my_nodes_3d)
    my_points_ = numpy.array(tuple(my_nodes_2d.values()))
    points_ = my_points_  # [:, [1, 0]]  # numpy.array(tuple(nodes_.values()))

    # ------------------
    # Generate split and plot it (without calling pyplot.show())
    # ------------------
    my_split_ = CentroidPolygons({f"Ch_{i_}": point for i_, point in enumerate(points_)},
                                 min_nodes=1, k=(5, 3))

    pyplot.figure()
    my_split_.plot(edge_color="black", line_width=3)

    # ------------------
    # Plot electrodes
    # ------------------
    nodes_3d = my_nodes_3d
    nodes_ = project_head_shape(electrode_positions=nodes_3d)

    groups = dict()
    for node_3d, (name, node_2d) in zip(nodes_3d.values(), nodes_.items()):
        # Get the region of the current node
        group = my_split_.coord_to_ch_group(CartesianCoordinates(node_3d), verbose=False)

        # Store it
        if group not in groups:
            groups[group] = dict()

        groups[group][name] = node_2d

    for nodes_ in groups.values():
        plot_2d_projection(electrode_positions=nodes_, annotate=False, s=150, font_size=18)

    pyplot.xlim((-1.7, 1.7))
    pyplot.ylim((-1.4, 1.4))

    pyplot.show()


if __name__ == "__main__":
    main()
