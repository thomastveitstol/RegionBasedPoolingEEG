import copy
import dataclasses
from typing import Dict, List, Optional, Tuple, Union
import warnings

from mne.transforms import _cart_to_sph, _pol_to_cart
import matplotlib
from matplotlib import pyplot
import numpy

from src.utils import CartesianCoordinates, PolarCoordinates, to_cartesian


# --------------------
# Classes for Node, Edge and PolygonGraph
# --------------------
@dataclasses.dataclass(frozen=True)
class Node:
    """
    Node class. This is meant to be used as vertices in PolygonGraph and Edge objects
    """

    position: Tuple[float, float]
    name: str = "X"


class Edge:
    """
    Class for defining an Edge between two nodes of type Node
    """

    __slots__ = "_node_1", "_node_2"

    def __init__(self, node_1: Node, node_2: Node):
        # Input check
        if not isinstance(node_1, Node) or not isinstance(node_2, Node):
            raise TypeError(f"Expected node 1 and node 2 to be of type {type(Node).__name__}, but found "
                            f"{type(node_1)} and {type(node_2)}")
        if node_1 == node_2:
            warnings.warn("The nodes passed were identical. An edge defined by two equal nodes is probably wrong")

        # -------------------
        # Set the nodes of the edge
        # -------------------
        self._node_1 = node_1
        self._node_2 = node_2

    def __eq__(self, other: 'Edge') -> bool:
        """
        __eq__ method. If A and B are two nodes, then the edge AB should be regarded as equal to BA.
        Args:
            other: An edge, which we want to compare with

        Returns: True if the edge is the same as self, False otherwise
        Examples:
            >>> my_n1 = Node((3.8, 2.4), "A")
            >>> my_n2 = Node((3.8, 2.4), "B")
            >>> Edge(my_n1, my_n2) == Edge(my_n1, my_n2)
            True
            >>> Edge(my_n1, my_n2) == Edge(my_n2, my_n1)
            True
            >>> Edge(my_n1, my_n2) == Edge(my_n1, Node( (3.8, 2.4), "BB"))
            False
            >>> Edge(my_n1, my_n2) == Edge(my_n1, Node((3.8, 2.3), "B"))
            False
        """
        # Input check. If the input is not type Edge, False should be returned
        if not isinstance(other, Edge):
            return False

        # Check the nodes one by one
        return (self._node_1 == other.node_1 and self._node_2 == other.node_2) or \
            (self._node_1 == other.node_2 and self._node_2 == other.node_1)

    def __repr__(self) -> str:
        """
        Method for printing
        Returns: String for printing
        Examples:
            >>> my_n1 = Node((3.8, 2.4), "A")
            >>> my_n2 = Node((3.4, -1.8), "B")
            >>> Edge(my_n1, my_n2)
            Edge(node_1=Node(position=(3.8, 2.4), name='A'), node_2=Node(position=(3.4, -1.8), name='B'))
        """
        return f"{type(self).__name__}(node_1={self._node_1}, node_2={self._node_2})"

    def line_intersect(self, point: Tuple[float, float], angle: float,
                       ray: bool = False) -> Optional[Tuple[float, float]]:
        """
        Get the x-value of the point where a line intersects with the edge. If there is no intersection, None will be
        returned
        Args:
            point: Point of the line
            angle: Angle of the line
            ray: If True, the line is interpreted as a ray intead of a line.

        Returns: x-value of intersection. If there is no intersection, None will be returned instead
        Examples:
            >>> my_point = (-2, 3)
            >>> my_angle = 3*numpy.pi/4
            >>> my_e = Edge(node_1=Node((1, -2), "A"), node_2=Node((2, 2), "B"))
            >>> my_x, my_y = my_e.line_intersect(point=(3, -2), angle=3*numpy.pi/4)  # x should equal 7/5
            >>> my_x, round(my_y, 5)  # x should equal 7/5, y should equal -2/5
            (1.4, -0.4)
            >>> my_e = Edge(node_1=Node((2, 2), "B"), node_2=Node((1, -2), "A"))
            >>> my_x, my_y = my_e.line_intersect(point=(3, -2), angle=3*numpy.pi/4)
            >>> my_x, round(my_y, 5)  # Output should be the same
            (1.4, -0.4)
            >>> my_e = Edge(node_1=Node((2, 2), "B"), node_2=Node((1, -2), "A"))
            >>> type(my_e.line_intersect(point=(3, -2), angle=3*numpy.pi/4+numpy.pi, ray=True))
            <class 'NoneType'>
            >>> my_e = Edge(node_1=Node((2, 2), "B"), node_2=Node((-1, 1), "A"))
            >>> my_x, my_y = my_e.line_intersect(point=(3, -2), angle=3*numpy.pi/4)
            >>> round(my_x, 5), round(my_y, 5)  # x should equal -1/4, y should equal 5/4
            (-0.25, 1.25)
            >>> my_e = Edge(node_1=Node((-1, -1), "A"), node_2=Node((1, -2), "B"))
            >>> type(my_e.line_intersect(point=(3, -2), angle=3*numpy.pi/4))  # Intersection not on the edge
            <class 'NoneType'>
            >>> n1 = Node(name='A', position=(-0.16171226930676674, 0.1374547696536207))
            >>> n2 = Node(name='C', position=(-0.16171226930676674, -0.1374547696536207))
            >>> my_e = Edge(n1, n2)
            >>> my_e.line_intersect(point=(0, 0), angle=0)
            (-0.16171226930676674, 0.0)
            >>> # The following should have given an intercept, but due to numerical issues it does not happen
            >>> my_vertices = (Node(position=(0.6232737552198611, -1.2245340963445983e-08), name='P1'),
            ...                Node(position=(0.9238181548922769, -1.823756307084423e-08), name='Q0'))
            >>> my_c = (0.7738704441194271, -7.961377836107137e-09)
            >>> my_angle = 4.712388978979588
            >>> Edge(*my_vertices).line_intersect(point=my_c, angle=my_angle, ray=True)
        """
        # -------------------------
        # Using formula for linear graphs
        # y_i = a_i(x - x_i) + y_i|x_i
        # -------------------------
        # Line
        line_a = numpy.tan(angle)  # Slope
        line_b = point[1]  # intersection with x=0 (after shifting x)
        line_x = point[0]  # shift in x

        # Edge (select the point of smallest x-value)
        edge_point_1 = self._node_1 if self._node_1.position[0] < self._node_2.position[0] else self._node_2
        edge_point_2 = self._node_1 if edge_point_1 == self._node_2 else self._node_2

        if edge_point_2.position[0] != edge_point_1.position[0]:
            edge_a = (edge_point_2.position[1] - edge_point_1.position[1]) / \
                     (edge_point_2.position[0] - edge_point_1.position[0])  # a = delta_y / delta_x
            edge_b = edge_point_1.position[1]  # intersection with x=0 (after shifting x)
            edge_x = edge_point_1.position[0]  # shift in x

            # -------------------------
            # Apply the formula which can  be found
            # by setting y_1 = y_2 and solving for x:
            #
            # x = (a_1*x_1 - a_2*x_2 + b_2 - b_1) / (a_1 - a_2)
            # -------------------------
            x_intersect = (line_a * line_x - edge_a * edge_x + edge_b - line_b) / (line_a - edge_a)
            y_intersect = line_a * (x_intersect - line_x) + line_b

            # If intersection happens on the wrong end of line and strict angle is True, the intersection is not valid
            if ray and numpy.cos(angle) * (x_intersect - line_x) < 0:
                return None

            # If intersection is on the edge, return x and y value, otherwise return None
            return (x_intersect, y_intersect) if edge_x < x_intersect < edge_point_2.position[0] else None
        else:
            # This is run in case the slope is infinite. We set the x-value to the only possible x-value, and check if
            # the y-value of the intersection is on the edge
            x_intersect = edge_point_1.position[0]
            y_intersect = line_a * (x_intersect - line_x) + line_b

            # If intersection happens on the wrong end of line and strict angle is True, the intersection is not valid
            if ray and numpy.cos(angle) * (x_intersect - line_x) < 0:
                return None

            # If intersection is on the edge, return x and y value, otherwise return None
            y_positions = edge_point_1.position[1], edge_point_2.position[1]
            return (x_intersect, y_intersect) if min(y_positions) < y_intersect < max(y_positions) else None

    # -------------
    # Properties
    # -------------
    @property
    def node_1(self) -> Node:
        return self._node_1

    @property
    def node_2(self) -> Node:
        return self._node_2


class PolygonGraph:
    """
    Class for defining polygons. They contain nodes in an ordered tuple.
    """

    __slots__ = "_nodes"

    def __init__(self, nodes: Tuple[Node, ...]):
        """
        Initialise.
        Args:
            nodes: The nodes of the Polygon. Note that the tuple is ordered, and that the (i+1)-th element will be the
                'next' element of the i-th element. The 'next' element of the final element is the first element
                (cyclic)
        """
        # --------------------
        # Set linked nodes
        # --------------------
        self._nodes = nodes

    def line_intersection(self, point: Tuple[float, float], angle: float,
                          ray: bool = False) -> Optional[Tuple[Tuple[Edge, ...], Tuple[Tuple[float, float], ...]]]:
        """
        Given a line, this method returns the edges which it intersects, as well as the coordinates of intersection. If
        there is no intersection, None will be returned.
        Args:
            point: Point of a line
            angle: Angle of the line
            ray: If True, the line is interpreted as a ray intead of a line.

        Returns: The edges of intersection, and their corresponding x-values of intersection. If there is no
            intersection, None will be returned instead
        Examples:
            >>> # Define nodes, edges and the PolygonGraph object
            >>> n1 = Node((-1, 1), "A")
            >>> n2 = Node((2, 2), "B")
            >>> n3 = Node((1, -2), "C")
            >>> my_polygon = PolygonGraph(nodes=(n1, n2, n3))
            >>> # Define the line
            >>> my_point = (1, 1)
            >>> my_angle = 3*numpy.pi/4
            >>> # Test if the intersections are as expected when ray is True
            >>> my_edges, my_intersections = my_polygon.line_intersection(point=my_point, angle=my_angle, ray=True)
            >>> my_edges  # doctest: +NORMALIZE_WHITESPACE
            (Edge(node_1=Node(position=(-1, 1), name='A'), node_2=Node(position=(2, 2), name='B')),)
            >>> # Test if the intersections are as expected
            >>> my_point = (3, -2)
            >>> my_edges, my_intersections = my_polygon.line_intersection(point=my_point, angle=my_angle)
            >>> my_edges  # doctest: +NORMALIZE_WHITESPACE
            (Edge(node_1=Node(position=(-1, 1), name='A'), node_2=Node(position=(2, 2), name='B')),
             Edge(node_1=Node(position=(2, 2), name='B'), node_2=Node(position=(1, -2), name='C')))
            >>> round(my_intersections[0][0], 5), round(my_intersections[0][1], 5), len(my_intersections[0])
            (-0.25, 1.25, 2)
            >>> round(my_intersections[1][0], 5), round(my_intersections[1][1], 5), len(my_intersections[1])
            (1.4, -0.4, 2)
            >>> len(my_intersections)
            2
            >>> my_point = (-0.09480746154814013, -0.04746348798130559)
            >>> my_angle = 3.583607130749073
            >>> n1 = Node(name='A', position=(-0.16171226930676674, 0.1374547696536207))
            >>> n2 = Node(name='B', position=(0.16171226930676674, 0.1374547696536207))
            >>> n3 = Node(name='C', position=(0.16171226930676674, -0.1364814536786759))
            >>> n4 = Node(name='D', position=(-0.16171226930676674, -0.1364814536786759))
            >>> len(PolygonGraph((n1, n2, n3, n4)).line_intersection(point=my_point, angle=my_angle))
            2
            >>> # If the point is not inside the polygon, two points are found when ray=True, and an error is NOT raised
            >>> my_polygon.line_intersection(point=(3, -2),
            ...                              angle=3*numpy.pi/4, ray=True)  # doctest: +NORMALIZE_WHITESPACE
            ((Edge(node_1=Node(position=(-1, 1), name='A'), node_2=Node(position=(2, 2), name='B')),),
             ((-0.24999999999999942, 1.2500000000000004),))
        """
        # Initialise lists
        intersections: List[Tuple[float, float]] = []  # Will contain coordinates
        edge_intersections: List[Edge] = []  # Will contain Edge objects

        # Checking all edges for intersection
        for edge in self.edges:
            # Get the intersection of the line
            intersection = edge.line_intersect(point=point, angle=angle, ray=ray)

            # If it is not None, store it
            if intersection is not None:
                intersections.append(intersection)
                edge_intersections.append(edge)

        # If no intersections are found, return None. For the class' intended and internal use cases, this should not
        # happen. Consider therefore to raise an error, or at least, a warning
        if not intersections:
            warnings.warn("No intersections were found, which is likely due to an error")
            return None

        # If the polygon is non-convex, too many intersections may have been found
        if len(intersections) != 1 and ray:
            # Select the edge closest to the point
            idx = numpy.argmin(numpy.linalg.norm(numpy.array(point) - numpy.array(intersections)))
            return (edge_intersections[idx],), (intersections[idx],)

        if len(intersections) != 2 and not ray:
            # This should never happen, if used correctly
            raise ValueError(f"The number of intersections with the line and polygon was unexpected "
                             f"({len(intersections)} given ray set to {ray})")

        return tuple(edge_intersections), tuple(intersections)

    def split(self, point: Tuple[float, float], angle: float,
              new_node_names: Tuple[str, str] = ("X", "X")) -> Tuple['PolygonGraph', 'PolygonGraph']:
        """
        Method for splitting a PolygonGraph into two, given a line.
        Args:
            point: Point of the splitting line
            angle: Angle of the splitting line
            new_node_names: Names of the node created in the split

        Returns: Two PolygonGraphs, which are made by splitting the original PolygonGraph
        Examples:
            >>> n1, n2, n3 = Node((-1, 1), "A"), Node((2, 2), "B"), Node((1, -2), "C")
            >>> my_polygon = PolygonGraph(nodes=(n1, n2, n3))
            >>> my_red_polygon, my_blue_polygon = my_polygon.split(point=(3, -2), angle=3*numpy.pi/4,
            ...                                                    new_node_names=("E", "F"))
            >>> my_reds = my_red_polygon.nodes
            >>> my_reds  # doctest: +NORMALIZE_WHITESPACE
            (Node(position=(-1, 1), name='A'), Node(position=(-0.24999999999999942, 1.2500000000000004), name='E'),
             Node(position=(1.4, -0.39999999999999947), name='F'), Node(position=(1, -2), name='C'))
            >>> my_blues = my_blue_polygon.nodes
            >>> my_blues  # doctest: +NORMALIZE_WHITESPACE
            (Node(position=(-0.24999999999999942, 1.2500000000000004), name='E'), Node(position=(2, 2), name='B'),
             Node(position=(1.4, -0.39999999999999947), name='F'))
        """
        # Get the edges and position of intersection
        edges, positions = self.line_intersection(point=point, angle=angle)

        assert len(edges) == len(positions) == 2  # Test if there are only two intersections

        # Get the first node of the edges intersected, in order (determined by the ordering of the nodes)
        first_split_node, second_split_node = (edges[0].node_1, edges[1].node_1) \
            if self._nodes.index(edges[0].node_1) < self._nodes.index(edges[1].node_1) \
            else (edges[1].node_1, edges[0].node_1)  # Consider to make this a function

        # Get the coordinates of nodes created by the line intersections
        node_intersect_position_1, node_intersect_position_2 = (positions[0], positions[1]) \
            if self._nodes.index(edges[0].node_1) < self._nodes.index(edges[1].node_1) else (positions[1], positions[0])

        # Make a copy of the node set, and convert it to a list
        nodes = list(self._nodes)

        # Insert the new nodes
        new_node_1 = Node(position=node_intersect_position_1, name=new_node_names[0])
        new_node_2 = Node(position=node_intersect_position_2, name=new_node_names[1])
        nodes.insert(self._nodes.index(first_split_node) + 1, new_node_1)
        nodes.insert(self._nodes.index(second_split_node) + 2, new_node_2)

        # --------------------
        # Split into blue and red
        # --------------------
        red_nodes = tuple(node for node in nodes if (nodes.index(node) <= nodes.index(new_node_1)) or
                          nodes.index(new_node_2) <= nodes.index(node))
        blue_nodes = tuple(node for node in nodes if
                           nodes.index(new_node_1) <= nodes.index(node) <= nodes.index(new_node_2))

        return PolygonGraph(red_nodes), PolygonGraph(blue_nodes)

    def multi_split(self, point: Tuple[float, float], angles: Tuple[float, ...]) -> Tuple['PolygonGraph', ...]:
        """

        Args:
            point: Point of the splitting ray
            angles: Angles of the splitting rays

        Returns: Multiple PolygonGraphs, which are made by splitting the original PolygonGraph
        Examples:
            >>> n1, n2, n3 = Node((1, -2), "C"), Node((2, 2), "B"), Node((-1, 1), "A")
            >>> my_polygon = PolygonGraph(nodes=(n1, n2, n3))
            >>> my_polygons = my_polygon.multi_split(point=(1, .5), angles=(3*numpy.pi/4, 3*numpy.pi/2, 1.9*numpy.pi))
            >>> tuple(node.name for node in my_polygons[0].nodes)  # type: ignore[attr-defined]
            ('P0', 'A', 'Q0', 'Centroid')
            >>> my_polygons[0].nodes[-1]
            Node(position=(1, 0.5), name='Centroid')
            >>> tuple(node.name for node in my_polygons[1].nodes)  # type: ignore[attr-defined]
            ('P1', 'C', 'Q1', 'Centroid')
            >>> tuple(node.name for node in my_polygons[2].nodes)  # type: ignore[attr-defined]
            ('P2', 'B', 'Q2', 'Centroid')
            >>> # An error may be raised if the method is not used properly (e.g. the point is not within the polygon)
            >>> n1, n2, n3 = (Node((0.9975, -0.124), 'P0'), Node((0.997, 0.331), 'Q0'),
            ...               Node((-0.036, -0.086), name='Centroid'))
            >>> PolygonGraph((n1, n2, n3)).multi_split(point=(-0.559, -0.729), angles=(1.308, 4.15, 5.71))
            Traceback (most recent call last):
            ...
                edges0, pos0 = self.line_intersection(point=point, angle=angle0, ray=True)
            TypeError: cannot unpack non-iterable NoneType object
            >>> v1, v2, v3, v4, v5 = (Node((-1.0, -0.89), 'P2'), Node((-1.0, -1.0), 'D'), Node((1.0, -1.0), 'C'),
            ...                       Node((1.0, -0.49), 'Q2'), Node((-0.0, -0.0), 'Centroid'))
            >>> my_c = (0.1, -0.64)
            >>> my_angles = (5.453483826071691, 0.5824221302166528, 3.1600788986843007)
            >>> p = PolygonGraph(nodes=(v1, v2, v3, v4, v5)).multi_split(point=my_c, angles=my_angles)
            >>> {node.name: (round(node.position[0], 2), round(node.position[1], 2))  # type: ignore[attr-defined]
            ...  for node in p[0].nodes}
            {'P0': (0.43, -1.0), 'C': (1.0, -1.0), 'Q2': (1.0, -0.49), 'Q0': (0.61, -0.3), 'Centroid': (0.1, -0.64)}
            >>> v1 = Node(position=(0.17302182658213516, -1.4852762314488022), name='P0')
            >>> v2 = Node(position=(1.2650649854725826, -1.4852762314488022), name='C')
            >>> v3 = Node(position=(1.2650649854725826, -1.3143802515361587), name='Q1')
            >>> v4 = Node(position=(0.3805365459575105, -0.9473614181782168), name='Q0')
            >>> v5 = Node(position=(0.08513037725592665, -1.2751828973542219), name='Centroid')
            >>> my_point = (0.47352129669517173, -1.2959547292171165)
            >>> my_angles = (5.570313234392192, 4.100741661544475)
            >>> _ = PolygonGraph((v1, v2, v3, v4, v5)).multi_split(my_point, my_angles)
            >>> # Example when the same edge is hit
            >>> my_point = (-0.39020995745450604, 0.9482026508720987)
            >>> my_angles = (0.5642341263009003, 5.219830772540187)
            >>> v0 = Node(position=(-0.7092074235900993, 0.9702220394403069), name='P0')
            >>> v1 = Node(position=(-0.701838856603698, 0.9617686295757834), name='Centroid')
            >>> v2 = Node(position=(-0.4888862512600215, 0.6474576858796685), name='P1')
            >>> v3 = Node(position=(-0.45048082508278914, 0.7130209221865673), name='Centroid')
            >>> v4 = Node(position=(0.05357220693019565, 1.4852762314488022), name='P0')
            >>> v5 = Node(position=(-0.04969157096064444, 1.4852762314488022), name='Q0')
            >>> v6 = Node(position=(-0.482089444345261, 1.0798237336771481), name='Centroid')
            >>> nodes_ = {'Ch_44': (-0.2606569, 1.02023478), 'Ch_49': (-0.51976302, 0.87617052)}
            >>> child_split = PolygonGraph((v0, v1, v2, v3, v4, v5, v6)).multi_split(my_point, my_angles)
            >>> from src.data.line_separation.centroid_polygons import _place_node_in_polygon  # noqa
            >>> _place_node_in_polygon(nodes=nodes_, polygons=child_split)
            {1: {'Ch_44': (-0.2606569, 1.02023478)}, 0: {'Ch_49': (-0.51976302, 0.87617052)}}
        """
        # Loop through all angle pairs (defining the region)
        polygon_graphs: List['PolygonGraph'] = list()
        for i, (angle0, angle1) in enumerate(zip(angles, angles[1:]+(angles[0],))):
            # Get the edges and position of intersection
            edges0, pos0 = self.line_intersection(point=point, angle=angle0, ray=True)
            edges1, pos1 = self.line_intersection(point=point, angle=angle1, ray=True)

            # Verify that there is only one intersection
            assert len(edges0) == len(pos0) == len(edges1) == len(pos1) == 1

            # Tuples of length 1 is not necessary
            edges0, pos0 = edges0[0], pos0[0]
            edges1, pos1 = edges1[0], pos1[0]

            # Get the node of the edges intersected, in order (determined by the ordering of the nodes)
            first_split_node, second_split_node = edges0.node_1, edges1.node_2

            # Get the coordinates of nodes created by the line intersections
            node_intersect_position_1, node_intersect_position_2 = pos0, pos1

            # Make a copy of the node set, and convert it to a list
            nodes = list(copy.deepcopy(self._nodes))

            # Make the new nodes
            new_edge_node_1 = Node(position=node_intersect_position_1, name=f"P{i}")
            new_edge_node_2 = Node(position=node_intersect_position_2, name=f"Q{i}")
            new_centroid = Node(position=point, name="Centroid")

            # Insert the new nodes
            if edges0 == edges1 and \
                    (numpy.linalg.norm(numpy.array(node_intersect_position_2) - numpy.array(first_split_node.position))
                     < numpy.linalg.norm(numpy.array(node_intersect_position_1) - numpy.array(first_split_node.position)
                                         )):
                nodes.insert(nodes.index(first_split_node) + 1, new_edge_node_1)
                nodes.insert(nodes.index(first_split_node)+1, new_edge_node_2)
                nodes.insert(nodes.index(first_split_node)+2, new_centroid)
            else:
                nodes.insert(nodes.index(first_split_node) + 1, new_edge_node_1)
                nodes.insert(nodes.index(second_split_node), new_edge_node_2)
                nodes.insert(nodes.index(second_split_node), new_centroid)

            # --------------------
            # Store split
            # --------------------
            i0 = nodes.index(new_edge_node_1)
            i1 = nodes.index(new_centroid)

            if i0 < i1:
                polygon_nodes = nodes[i0:(i1+1)]
            else:
                polygon_nodes = nodes[i0:] + nodes[:(i1+1)]
            polygon_graphs.append(PolygonGraph(tuple(polygon_nodes)))

        return tuple(polygon_graphs)

    def plot(self, face_color: str = "random", edge_color: str = "darkblue", line_width: int = 2) -> None:
        """Plot the polygon"""
        # -------------
        # Get positions
        # -------------
        node_positions = tuple(node.position for node in self._nodes)

        # Get as x and y values for compatibility with pyplot.fill()
        x, y = zip(*node_positions)

        # -------------
        # Plotting
        # -------------
        if face_color == "random":
            # Sample color from colormap
            cmap = matplotlib.cm.get_cmap('YlOrBr')

            # Get face color
            face_color = cmap(numpy.random.randint(low=0, high=cmap.N//2))

        pyplot.fill(x, y, linewidth=line_width, facecolor=face_color, edgecolor=edge_color)

    # ------------
    # Properties
    # ------------
    @property
    def nodes(self) -> Tuple[Node, ...]:
        return self._nodes

    @property
    def edges(self) -> Tuple[Edge, ...]:
        """Generate edges from the nodes"""
        # Initialise edge list. All edges will be appended to this list
        edge_list: List[Edge] = []

        # Loop through nodes, and get both the i-th and the i+1-th element
        for node_0, node_1 in zip(self._nodes[:-1], self._nodes[1:]):
            # Append edge between the i-th and the i+1-th element to the set of edges
            edge_list.append(Edge(node_1=node_0, node_2=node_1))

        # Add a connection from the last element to the first element, and return as a tuple
        edge_list.append(Edge(node_1=self._nodes[-1], node_2=self._nodes[0]))
        return tuple(edge_list)


# --------------------
# Functions for projecting from 3D to 2D
# --------------------
def project_head_shape(electrode_positions: Dict[str, Union[PolarCoordinates, CartesianCoordinates, numpy.ndarray]]) \
        -> Dict[str, Tuple[float, float]]:
    """
    Function for projecting 3D points to 2D, as done in MNE for plotting sensor location.

    Most of this code was taken from the _auto_topomap_coordinates function, to obtain the same mapping as MNE. Link to
    this function can be found at (source code):
    https://github.com/mne-tools/mne-python/blob/9e4a0b492299d3638203e2e6d2264ea445b13ac0/mne/channels/layout.py#L633
    Args:
        electrode_positions: Dictionary of electrode positions. The keys are channel names, the values are coordinates.
            The coordinates can be of type PolarCoordinates, CartesianCoordinates or array-like. If array-like, the
            array is expected to contain cartesian coordinates, in the order x, y, z

    Returns: The 2D projection of the electrodes. Keys are channel names. Values are 2D-projection
    Examples:
        >>> import mne
        >>> my_positions = mne.channels.make_standard_montage(kind="GSN-HydroCel-129").get_positions()["ch_pos"]
        >>> tuple(project_head_shape(electrode_positions=my_positions).keys())[:3]
        ('E1', 'E2', 'E3')
        >>> tuple(project_head_shape(electrode_positions=my_positions).values())[:3]
        (array([0.07890224, 0.0752648 ]), array([0.05601906, 0.07102252]), array([0.03470422, 0.06856416]))
    """
    # ---------------------------
    # Handling varied inputs. No formal input
    # check per sé, as long as it can be
    # converted to a numpy array
    # ---------------------------
    if isinstance(tuple(electrode_positions.values())[0], (PolarCoordinates, CartesianCoordinates)):
        # Map to cartesian coordinates
        electrode_coords = numpy.array([to_cartesian(electrode).coordinates
                                        for electrode in electrode_positions.values()])
    else:
        # Get positions to numpy array
        electrode_coords = electrode_positions.values()
        electrode_coords = numpy.array([coord for coord in electrode_coords])

    # ---------------------------
    # Apply the same steps as _auto_topomap_coordinates
    # from MNE.transforms
    # ---------------------------
    cartesian_coords = _cart_to_sph(electrode_coords)
    out = _pol_to_cart(cartesian_coords[:, 1:][:, ::-1])
    out *= cartesian_coords[:, [0]] / (numpy.pi / 2.)

    # Convert to Dict and return
    return {channel_name: projection_2d for channel_name, projection_2d in zip(electrode_positions, out)}


def plot_2d_projection(electrode_positions: Dict[str, Tuple[float, float]], annotate: bool = True,
                       s: Optional[float] = None, font_size: int = 13):
    # Extract positions as x and y numpy arrays
    positions = electrode_positions.values()

    x = numpy.array([x for x, _ in positions])
    y = numpy.array([y for _, y in positions])

    # --------------
    # Plotting
    # --------------
    pyplot.scatter(x, y, s=s)

    # Annotate the channels, if desired
    if annotate:
        for x_pos, y_pos, channel_name in zip(x, y, electrode_positions):
            pyplot.text(x=x_pos, y=y_pos, s=channel_name, fontsize=font_size)

    pyplot.ylim(min(y) - 0.2*abs(min(y)), max(y) + 0.2*abs(max(y)))
