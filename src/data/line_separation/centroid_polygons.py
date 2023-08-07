"""
Class and functions generating separating lines. To split a group of electrodes, the centroid is calculated. From the
centroid, k number of circle sectors are created, which union forms a circle and intersection is the empty set. All
regions contain 1/k th number of electrodes
"""
from typing import Dict, Optional, Tuple, Union

import numpy
from shapely.geometry import Point, Polygon
import sympy

from src.data.line_separation.region_split_base import RegionsBase
from src.data.line_separation.utils import Node, PolygonGraph, project_head_shape
from src.utils import CartesianCoordinates, PolarCoordinates, ChGroup


class CentroidPolygons(RegionsBase):

    __slots__ = "_centroid", "_separating_angles", "_polygon"

    candidate_settings = ({"k": (3, 3, 3, 3, 3)}, {"k": (4, 2, 4, 3, 4, 3)}, {"k": (2, 4, 2, 4, 2, 4)},
                          {"k": (8,)}, {"k": (7,)}, {"k": (6,)}, {"k": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)},
                          {"k": (3, 2, 3, 2, 3, 2)}, {"k": (2, 3, 2, 3, 2, 3)}, {"k": (3, 4, 2, 3, 4, 2)})

    def __init__(self, nodes: Dict[str, Tuple[float, float]], k: Tuple[int, ...], min_nodes: Optional[int] = 1,
                 add_node_noise: bool = True, _polygon: Optional[PolygonGraph] = None):
        """
        Initialise
        Args:
            nodes: Nodes to fit the split on
            k: number of regions
        Examples:
            >>> my_split = CentroidPolygons(
            ...     {f"Ch_{i_}": tuple(numpy.random.uniform(-1, 1, size=2))  # type: ignore[attr-defined]
            ...      for i_ in range(100)}, k=(10, 3, 3, 7))
        """
        min_nodes = 1 if min_nodes is None else min_nodes

        # -----------------------
        # Store nodes and edges of the
        # graph/polygons defining the region
        # -----------------------
        # Get or define the polygon graph of the entire region, before splitting
        if _polygon is None:
            # Maybe add a small noise to the nodes. This is to prevent aligned electrodes
            if add_node_noise:
                nodes = {ch_name: tuple(numpy.array(pos)+numpy.random.uniform(low=-1e-8, high=1e-8))
                         for ch_name, pos in nodes.items()}

            # The polygon of the first split is a box containing all electrodes
            x_values = tuple(x for x, _ in nodes.values())
            y_values = tuple(y for _, y in nodes.values())

            # Ordering is important, must be counterclockwise
            node_a = Node(name="D", position=(min(x_values) - 1e-3, min(y_values) - 1e-3))
            node_b = Node(name="C", position=(max(x_values) + 1e-3, min(y_values) - 1e-3))
            node_c = Node(name="B", position=(max(x_values) + 1e-3, max(y_values) + 1e-3))
            node_d = Node(name="A", position=(min(x_values) - 1e-3, max(y_values) + 1e-3))

            # Set the polygon
            self._polygon = PolygonGraph(nodes=(node_a, node_b, node_c, node_d))
        else:
            self._polygon = _polygon

        # -----------------------
        # Store centroid and angles of
        # separating lines
        # -----------------------
        midpoint = _compute_centroid(nodes=nodes)

        # Check if the centroid is contained in the Polygon
        if not Polygon(tuple(node.position for node in self._polygon.nodes)).contains(Point(midpoint)):
            midpoint = Polygon(tuple(node.position for node in self._polygon.nodes)).representative_point()
            midpoint = (midpoint.x, midpoint.y)

        # Check if the polygon intersects itself. Should never happen though
        if not Polygon(tuple(node.position for node in self._polygon.nodes)).is_simple:
            raise ValueError("Polygon contains intersection, this should never happen")

        self._centroid = midpoint
        self._separating_angles = _compute_separating_angles(nodes=nodes, centroid=self._centroid, k=k[0])

        # -----------------------
        # Store polygons of the regions
        # -----------------------
        try:
            self._child_polygons = self._polygon.multi_split(point=self._centroid, angles=self._separating_angles)
        except TypeError:
            # There are occasions where the numerical precision of the Edge().line_intersect is too poor
            raise NumericalError

        # -----------------------
        # Store node names and positions
        # in their correct color dict
        # -----------------------
        placed_nodes = _place_node_in_polygon(nodes=nodes, polygons=self._child_polygons)

        # Maybe regret this recursion
        if not (len(placed_nodes) == len(self._child_polygons) == k[0]):
            # This happens due to a flaw in the method. Not an error per sé, as the problem is known and will get dealt
            # with
            raise RegretRecursionError

        # -----------------------
        # (Maybe) split children groups
        # -----------------------
        self._children_split: Dict[int, Optional['CentroidPolygons']] = dict()  # integer keys somewhat inelegant?
        try:
            for region, polygon_nodes in placed_nodes.items():
                if len(k) == 1 or len(polygon_nodes) // k[1] < min_nodes:
                    self._children_split[region] = None
                elif any(len(polygon_nodes) // k[1] < min_nodes for polygon_nodes in placed_nodes.values()):
                    self._children_split[region] = None
                else:
                    # The method itself is fundamentally flawed in some non-convex cases. A pragmatic solution to this
                    # is currently to simply catch whenever this happens, and regret the
                    self._children_split[region] = CentroidPolygons(nodes=polygon_nodes, k=k[1:],
                                                                    _polygon=self._child_polygons[region],
                                                                    min_nodes=min_nodes)
        except (RegretRecursionError, IntersectingPolygonError):
            for region in placed_nodes:
                self._children_split[region] = None

    # -------------------
    # Methods for placing electrodes into region/ChGroup
    # -------------------
    def coord_to_ch_group(self, coordinates: Union[CartesianCoordinates, PolarCoordinates],
                          verbose: bool = False) -> ChGroup:
        # Get 2D-projection
        position = tuple(project_head_shape(electrode_positions={"": coordinates}).values())[0]

        # Place using 2d-method
        return self._coord_2d_to_ch_group(coordinates=position, verbose=verbose)

    def _coord_2d_to_ch_group(self, coordinates: Tuple[float, float], verbose: bool = False,
                              _color_sequence: Tuple[int, ...] = tuple()) -> ChGroup:
        """
        Method for placing a coordinate in a region
        Args:
            coordinates: Position of the point as a tuple (x, y)
            verbose: To print the derivation or not.
            _color_sequence: Color sequence (using integers to represent colors). Used internally for recursion

        Returns: The region (ChGroup) the input coordinate belongs to

        """
        # Place the point, and add it to the sequence
        node_color = _place_single_node_in_polygon(node_position=coordinates, polygons=self._child_polygons)
        color_sequence = _color_sequence + (node_color,)

        if verbose:
            print(f"1: {color_sequence}")

        # Maybe initiate child split
        channel_group = None
        for polygon_id, child_split in self._children_split.items():
            if node_color == polygon_id and child_split is not None:
                channel_group = child_split._coord_2d_to_ch_group(coordinates=coordinates, verbose=verbose,
                                                                  _color_sequence=color_sequence)
                break

        # Return if and only if the node split is the final one
        if all(child_split is None for child_split in self._children_split.values()):
            if verbose:
                print(f"2: {color_sequence}")

            # A clever trick using prime numbers. The number may get very large though.
            # It essentially assigns an integer from the color sequence, which is guaranteed to be unique for that very
            # color sequence. If the color sequence is e.g. (3, 2, 1, 4, 8), then the integer assigned is
            # 2^3 * 3^2 * 5^1 * 7^4 * 11^8. The integer is then passed to ChGroup() to have the correct type
            primes = _first_n_primes(n=len(color_sequence))
            return ChGroup(int(numpy.prod(tuple(prime**col for col, prime in zip(color_sequence, primes)))))

        return channel_group

    # -------------------
    # Methods for plotting
    # -------------------
    def plot_polygons(self, face_color: str = "random", edge_color: str = "darkblue", line_width: int = 2) -> None:
        # ------------------
        # Initiate plot children polygons,
        # if they exist
        # ------------------
        for child_split in self._children_split.values():
            if child_split is not None:
                child_split.plot_polygons(face_color=face_color, edge_color=edge_color, line_width=line_width)

        # ------------------
        # Plot if and only if there are no children
        # nodes (the current node is a terminal node)
        # ------------------
        if all(child_split is None for child_split in self._children_split.values()):
            for child_polygon in self._child_polygons:
                child_polygon.plot(face_color=face_color, edge_color=edge_color, line_width=line_width)

    def plot(self, face_color: str = "random", edge_color: str = "darkblue", line_width: int = 2) -> None:
        """Plots the region. No difference from .plot_polygon() method"""
        self.plot_polygons(face_color=face_color, edge_color=edge_color, line_width=line_width)


class CentroidPolygonsConf2(CentroidPolygons):
    candidate_settings = ({"k": (3, 3, 3, 3, 3, 3, 3, 3)}, {"k": (4, 2, 4, 3, 4, 3)}, {"k": (2, 4, 2, 4, 2, 4, 2, 4)},
                          {"k": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)},
                          {"k": (3, 2, 3, 2, 3, 2)}, {"k": (2, 3, 2, 3, 2, 3)}, {"k": (3, 4, 2, 3, 4, 2)})


# --------------------
# Errors
# --------------------
class RegretRecursionError(Exception):
    """
    Error class which is raised whenever we would like to regret making a recursion
    """
    pass


class IntersectingPolygonError(Exception):
    """
    Error class which is raised when a polygon intersects itself when it should not. Ideally this should never happen,
    but due to time constraints for runs, this will do for a temporary solution
    """
    pass


class NumericalError(Exception):
    """
    Error class which is raised when an error happened due to numerical issues
    """


# --------------------
# Functions
# --------------------
def _first_n_primes(n: int) -> Tuple[int, ...]:
    """
    Get the first n primes
    Args:
        n: Number of primes to return

    Returns: The first n primes
    Examples:
        >>> _first_n_primes(5)
        (2, 3, 5, 7, 11)
        >>> all(len(_first_n_primes(i)) == i for i in range(1, 100))  # type: ignore[attr-defined]
        True
    """
    return tuple(sympy.primerange(sympy.prime(n)+1))


def _place_single_node_in_polygon(node_position: Tuple[float, float], polygons: Tuple[PolygonGraph, ...]) -> int:
    """
    Place a single node in a polygon. todo: extreme similarities with the function below
    Args:
        node_position: x and y position of node to place in polygon
        polygons: Polygons, which the node will be attempted placed in

    Returns: int ID, the first polygon index the node is in.

    """
    # Try all polygons
    for i, polygon in enumerate(polygons):
        # If the node is contained in the current polygon, store it
        if Polygon(tuple(node.position for node in polygon.nodes)).contains(Point(node_position)):
            return i

    # Could not be placed in a polygon
    return -1


def _place_node_in_polygon(nodes: Dict[str, Tuple[float, float]],
                           polygons: Tuple[PolygonGraph, ...]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Function for placing nodes in polygons. The nodes will be placed in the first polygon it is in. If a node is not in
    any of the polygons, it will not be added to the output dict
    Args:
        nodes:
        polygons:

    Returns: Keys are index of polygon (polygon number, as by the ordering in the input tuple), and values are all the
        nodes placed in the corresponding polygon
    """
    placed_nodes: Dict[int, Dict[str, Tuple[float, float]]] = dict()  # making this a class? also, int keys?
    for node_name, node_position in nodes.items():
        # Try all polygons
        for i, polygon in enumerate(polygons):
            if not Polygon(tuple(node.position for node in polygon.nodes)).is_simple:
                raise IntersectingPolygonError

            # If the node is contained in the current polygon, store it
            if Polygon(tuple(node.position for node in polygon.nodes)).contains(Point(node_position)):
                # Maybe add the group as a key
                if i not in placed_nodes:
                    placed_nodes[i] = dict()

                # Place node
                placed_nodes[i][node_name] = node_position

                # No need to check the other polygons (unless you have overlapping regions, which we don't in this case)
                break

    return placed_nodes


def _compute_centroid(nodes: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Function for computing the centroid 2D position of nodes
    Args:
        nodes: Nodes where keys are names and values are 2d coordinates

    Returns: 2D position of centroid, (x, y)
    Examples:
        >>> my_nodes = {"Cz": (0.2, 0.1), "Cpz": (0.3, -50)}
        >>> _compute_centroid(nodes=my_nodes)
        (0.25, -24.95)
    """
    return tuple(numpy.mean(numpy.array(list(nodes.values())), axis=0))


def _compute_separating_angles(nodes: Dict[str, Tuple[float, float]], centroid: Tuple[float, float],
                               k: int) -> Tuple[float, ...]:
    """
    Get the angles for splitting into k regions
    Args:
        nodes: Nodes, with shape=(num_nodes, 2)
        centroid: Position of the centroid
        k: number of regions to split the group into

    Returns: The angles to use for separation into regions
    Examples:
        >>> numpy.random.seed(2)
        >>> my_theta = numpy.linspace(0, numpy.pi * 2, num=1001)
        >>> my_x = numpy.cos(my_theta)
        >>> my_y = numpy.sin(my_theta)
        >>> my_nodes = {f"Ch_{i}": (p0, p1) for i, (p0, p1) in enumerate(zip(my_x, my_y))}  # type: ignore[attr-defined]
        >>> my_centroid = (0., 0.)
        >>> my_angles = _compute_separating_angles(my_nodes, centroid=my_centroid, k=4)
        >>> # Some deviations from perfect circle is expected, due to num != inf
        >>> tuple(round(my_angle/numpy.pi, 3) for my_angle in my_angles)  # type: ignore[attr-defined]
        (0.872, 1.371, 1.873, 0.371)
        >>> len(my_angles)
        4
    """
    # Get the positions only, as numpy arrays
    node_positions = numpy.array(list(nodes.values()))

    # Set origin to centroid
    node_positions -= centroid

    # Randomly select a starting angle
    start_angle = numpy.random.uniform(0, 2*numpy.pi)

    # Compute angles with respect to start angle
    angles = numpy.mod(numpy.mod(numpy.arctan2(node_positions[:, 1], node_positions[:, 0]), 2*numpy.pi) - start_angle,
                       2*numpy.pi)

    # Sort angles (not the best implementation. But it is probably not significant anyway)
    sorted_angles = numpy.sort(numpy.insert(angles, 0, 0))

    # Make k partitions
    partitions = numpy.array_split(sorted_angles, indices_or_sections=k)

    # Compute separating angles
    separating_angles = [numpy.mod((p0[-1] + p1[0])/2+start_angle, 2*numpy.pi) for p0, p1 in zip(partitions[:-1],
                                                                                                 partitions[1:])]

    # Add the starting angle as separating angle
    separating_angles.insert(0, start_angle)

    return tuple(separating_angles)
