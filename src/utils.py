import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pickle

import numpy


# --------------------------------
# Class convenient to inherit from
# --------------------------------
class InputStore:
    """
    Inherit from this class to automatically store input arguments and keyword arguments in a dictionary called
    'inputs' (available as a property). Also default values will be included in the 'inputs' dict

    TODO: this does not work properly if using **kwargs
    Example:
        >>> class MyClass(InputStore):
        ...     def __init__(self, /, arg1, *, arg2=5, arg3=17):
        ...         pass
        >>> obj = MyClass("hello", arg3=57)
        >>> print(obj.inputs)
        {'arg1': 'hello', 'arg3': 57, 'arg2': 5}
    """

    __slots__ = "_inputs"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)

        # ------------------
        # Handle args and kwargs
        # ------------------
        num_input_args = len(args)
        arg_names = cls.__init__.__code__.co_varnames[1:num_input_args+1]  # not using self

        inputs = {name: value for name, value in zip(arg_names, args)}
        inputs.update(kwargs)

        # If no defaults are used, return object
        if cls.__init__.__code__.co_argcount - 1 == len(args) + len(kwargs):
            obj._inputs = inputs
            return obj

        # ------------------
        # Store defaults not overridden
        # ------------------
        default_values = cls.__init__.__defaults__  # Does not include pos only
        default_values = tuple() if default_values is None else default_values
        kw_default_values = cls.__init__.__kwdefaults__
        if kw_default_values is not None:
            default_values += tuple(kw_default_values.values())

        variables_with_defaults = cls.__init__.__code__.co_varnames[-len(default_values):]

        for var_name, var_value in zip(variables_with_defaults, default_values):
            if var_name in inputs:
                continue
            inputs[var_name] = var_value

        obj._inputs = inputs
        return obj

    def update_input_dict(self, key: str, value: Any) -> None:
        """Add or change the dict"""
        self._inputs[key] = value

    def delete_input(self, key: str) -> None:
        del self._inputs[key]

    @property
    def inputs(self) -> Dict[str, Any]:
        return self._inputs


# --------------------------------
# Defining convenient dataclasses
# --------------------------------
@dataclass(frozen=True)
class ChGroup:
    """Channel Group ID"""
    group_number: int


@dataclass(frozen=True)
class PolarCoordinates:
    """Polar Coordinates"""
    coordinates: Tuple[float, float, float]


@dataclass(frozen=True)
class CartesianCoordinates:
    """Cartesian Coordinates"""
    coordinates: Tuple[float, float, float]


# --------------------------------
# Mapping coordinates
# --------------------------------
def polar_to_cartesian(coordinates: PolarCoordinates) -> CartesianCoordinates:
    """
    Convert from polar coordinates to cartesian coordinates
    Args:
    coordinates: Polar coordinates of a single point, with the order (rho, theta, phi). Should satisfy the
        following: rho >= 0, 0 <= theta <= pi, 0 <= phi <= 2*pi
    Returns: Conversion to cartesian coordinates. The cartesian coordinates satisfy the following:
        -inf < x < inf, -inf < y < inf, -inf < z < inf
    Examples:
        >>> polar_to_cartesian(PolarCoordinates((1, numpy.pi/2, numpy.pi/2))).coordinates  # Some numerical errors
        (6.123233995736766e-17, 1.0, 6.123233995736766e-17)
        >>> polar_to_cartesian(coordinates=(1, numpy.pi/2, numpy.pi/2))  # type: ignore
        Traceback (most recent call last):
        ...
        TypeError: Expected type <class 'utils.PolarCoordinates'>, but received <class 'tuple'>
        >>> polar_to_cartesian(PolarCoordinates((-.4, numpy.pi/2, numpy.pi/2)))
        Traceback (most recent call last):
        ...
        ValueError: rho must be greater or equal to zero, but was rho=-0.4
        >>> polar_to_cartesian(PolarCoordinates((.4, 1.2*numpy.pi, numpy.pi/2)))
        Traceback (most recent call last):
        ...
        ValueError: theta must be in [0, pi], but was theta=3.7699111843077517
        >>> polar_to_cartesian(PolarCoordinates((.4, numpy.pi/3, -numpy.pi/5)))
        Traceback (most recent call last):
        ...
        ValueError: phi must be in [0, 2*pi), but was phi=-0.6283185307179586
    """
    # Check input type
    if not isinstance(coordinates, PolarCoordinates):
        raise TypeError(f"Expected type {PolarCoordinates}, but received {type(coordinates)}")

    # Extract polar coordinates
    rho, theta, phi = coordinates.coordinates

    # Check inputs
    if rho < 0:
        raise ValueError(f"rho must be greater or equal to zero, but was {rho=}")
    if theta < 0 or theta > numpy.pi:
        raise ValueError(f"theta must be in [0, pi], but was {theta=}")
    if phi < 0 or phi >= 2*numpy.pi:
        raise ValueError(f"phi must be in [0, 2*pi), but was {phi=}")

    # Convert to cartesian
    x = rho * numpy.cos(phi) * numpy.sin(theta)
    y = rho * numpy.sin(phi) * numpy.sin(theta)
    z = rho * numpy.cos(theta)

    # return as (x, y, z)
    return CartesianCoordinates((x, y, z))


def cartesian_to_polar(coordinates: CartesianCoordinates) -> PolarCoordinates:
    """
    Convert from cartesian coordinates to polar coordinates
    Args:
        coordinates: Cartesian coordinates of a single point, with the order (x, y, z)
    Returns: Conversion to polar coordinates (rho, theta, phi). The polar coordinates satisfy the following:
        rho >= 0, 0 <= theta <= pi,  0 <= phi <= 2*pi
    Examples:
        >>> cartesian_to_polar(CartesianCoordinates((.9, -.2, .5))).coordinates
        (1.0488088481701516, 1.0738638360156767, 6.064516361305644)
        >>> cartesian_to_polar(CartesianCoordinates((.9, .2, .5))).coordinates
        (1.0488088481701516, 1.0738638360156767, 0.21866894587394198)
        >>> cartesian_to_polar(CartesianCoordinates((1, -0.001, 1))).coordinates
        (1.4142139159264415, 0.7853984133973234, 6.282185307512919)
        >>> cartesian_to_polar(coordinates=(1, -0.001, 1))  # type: ignore
        Traceback (most recent call last):
        ...
        TypeError: Expected type <class 'utils.CartesianCoordinates'>, but received <class 'tuple'>
    """
    # Check input type
    if not isinstance(coordinates, CartesianCoordinates):
        raise TypeError(f"Expected type {CartesianCoordinates}, but received {type(coordinates)}")

    # Extract cartesian coordinates
    x, y, z = coordinates.coordinates

    # Convert to polar coordinates
    rho = numpy.sqrt(x**2 + y**2 + z**2)
    theta = numpy.arccos(z/rho)
    if x > 0:
        phi = numpy.arctan(y/x)
        if y < 0:
            phi += 2*numpy.pi
    elif x < 0:
        phi = numpy.arctan(y/x) + numpy.pi
    elif x == 0:
        if y > 0:
            phi = numpy.pi / 2
        elif y < 0:
            phi = 3*numpy.pi / 2
        else:
            warnings.warn(f"Cannot have an azimuthal angle for x = 0 and y = 0. Setting phi = -pi / 2")
            phi = -numpy.pi
            # raise ValueError(f"Cannot have an azimuthal angle for x = 0 and y = 0")
    else:
        raise ValueError(f"Combination of x = {x} and y = {y} not recognised")

    # return as the tuple (rho, theta, phi)
    return PolarCoordinates((rho, theta, phi))


def to_polar(coordinates: Union[PolarCoordinates, CartesianCoordinates]) -> PolarCoordinates:
    """
    Get polar coordinates. If the coordinates are cartesian, it is mapped to polar. If the coordinates are polar, the
    input is simply returned. If the type is neither cartesian nor polar coordinates, a TypeError is raised
    Args:
        coordinates: Either cartesian or polar coordinates
    Returns:
        Polar coordinates of the input
    Examples:
        >>> to_polar(coordinates=CartesianCoordinates((.9, -.2, .5)))
        PolarCoordinates(coordinates=(1.0488088481701516, 1.0738638360156767, 6.064516361305644))
        >>> to_polar(coordinates=PolarCoordinates((1., 1.5673, 3.1415)))
        PolarCoordinates(coordinates=(1.0, 1.5673, 3.1415))
        >>> to_polar(coordinates=(1., 1.5673, 3.1415))  # type: ignore   # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        TypeError: Expected coordinates of type <class 'utils.CartesianCoordinates'> or
            <class 'utils.PolarCoordinates'>, but received type <class 'tuple'>
    """
    if isinstance(coordinates, CartesianCoordinates):
        return cartesian_to_polar(coordinates)
    elif isinstance(coordinates, PolarCoordinates):
        return coordinates
    else:
        raise TypeError(f"Expected coordinates of type {CartesianCoordinates} or {PolarCoordinates}, but received"
                        f" type {type(coordinates)}")


def to_cartesian(coordinates: Union[PolarCoordinates, CartesianCoordinates]) -> CartesianCoordinates:
    """
    Get cartesian coordinates. If the coordinates are cartesian, the input is returned. If the coordinates are polar,
    it is mapped to cartesian. If the type is neither cartesian nor polar coordinates, a TypeError is raised
    Args:
        coordinates: Either cartesian or polar coordinates
    Returns: Cartesian coordinates of the input
    Examples:
        >>> to_cartesian(coordinates=CartesianCoordinates((.9, -.2, .5)))
        CartesianCoordinates(coordinates=(0.9, -0.2, 0.5))
        >>> to_cartesian(coordinates=PolarCoordinates((1., 1.5673, 3.1415)))
        CartesianCoordinates(coordinates=(-0.9999938835633805, 9.265302334838016e-05, 0.003496319671542502))
        >>> to_cartesian(coordinates=(.9, -.2, .5))  # type: ignore  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        TypeError: Expected coordinates of type <class 'utils.CartesianCoordinates'> or
            <class 'utils.PolarCoordinates'>, but received type <class 'tuple'>
    """
    if isinstance(coordinates, CartesianCoordinates):
        return coordinates
    elif isinstance(coordinates, PolarCoordinates):
        return polar_to_cartesian(coordinates)
    else:
        raise TypeError(f"Expected coordinates of type {CartesianCoordinates} or {PolarCoordinates}, but received"
                        f" type {type(coordinates)}")


# -------------------------------
# Dictionary utils
# -------------------------------
def append_values(append_dict: Dict[Any, Any],
                  main_dict: Optional[Dict[Any, List[Any]]] = None) -> Dict[Any, List[Any]]:
    """
    Append the values in 'append_dict' to the values in 'main_dict'. If 'main_dict' is None (or any other object
    which bool(main_dict) returns False), append_dict will be returned with values in a list (see Examples). Only the
    values which corresponding key exist in main_dict will be appended, and a KeyError will be raised if there exist a
    key in main_dict which does not exist in append_dict (this may be accepcted in the future).
    Args:
        main_dict: Main dict, which we want to append 'append_dict' to. If set to None, 'append_dict' will be returned,
            with its values in a list
        append_dict: The dictionary we want to append to main_dict

    Returns:
        A dictionary with the values of append_dict appended to the values of main_dict
    Examples:
        >>> append_values(append_dict={"a": 1, "b": 42, "c": 95})  # no main_dict passed (passed as None by default)
        {'a': [1], 'b': [42], 'c': [95]}
        >>> append_values(append_dict={"a": 76, "b": 36, "c": 180}, main_dict={})  # empty main_dict passed
        {'a': [76], 'b': [36], 'c': [180]}
        >>> append_values(append_dict={"a": 45, "b": 23, "c": 77}, main_dict={"a": [5, 4], "b": [3, 0], "c": [66, 88]})
        {'a': [5, 4, 45], 'b': [3, 0, 23], 'c': [66, 88, 77]}
        >>> my_dict = append_values(append_dict={"a": [.1, .2, .3], "b": [.0, -.1, -.2]}, main_dict=None)
        >>> my_dict = append_values(append_dict={"a": [.15, .25, .35], "b": [.02, -.12, -.22]}, main_dict=my_dict)
        >>> my_dict = append_values(append_dict={"a": [.16, .26, .36], "b": [.07, -.17, -.27]}, main_dict=my_dict)
        >>> my_dict  # doctest: +NORMALIZE_WHITESPACE
        {'a': [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.16, 0.26, 0.36]],
         'b': [[0.0, -0.1, -0.2], [0.02, -0.12, -0.22], [0.07, -0.17, -0.27]]}
        >>> numpy.mean(numpy.array(my_dict["a"]), axis=0)
        array([0.13666667, 0.23666667, 0.33666667])
        >>> # If there are keys in append_dict not present in main_dict, they will not be appended
        >>> append_values(append_dict={"a": 1, "b": 2, "c": 3}, main_dict={"a": [4, 3], "c": [6, 5]})
        {'a': [4, 3, 1], 'c': [6, 5, 3]}
        >>> # If there are keys in main_dict not present in append_dict, a KeyError is raised
        >>> append_values(append_dict={"a": 1, "c": 3}, main_dict={"a": [4, 3], "b": [7, 4], "c": [6, 5]})
        Traceback (most recent call last):
        ...
        KeyError: 'b'
    """
    # If bool(main_dict) returns False (such as None and empty dict {}), return append_dict with values in a list
    if not main_dict:
        return {key: [value] for key, value in append_dict.items()}
    else:
        # Loop through the items of main_dict, and append the values in append_dict
        return {key: main_value+[append_dict[key]] for key, main_value in main_dict.items()}


# -------------------------------
# Get metric dictionaries
# -------------------------------
def get_metrics(path: str) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Get all the metrics in a folder (e.g. train_history.pkl and val_history.pkl). All elements in the folder which has
    a name ending with '_history.pkl' is interpreted as a metric Dict to load
    Args:
        path: Path to folder

    Returns:
        A Dictionary with all metrics available. Keys are e.g. 'train', 'val'. The values are dictionaries with metric
            name (e.g. 'accuracy') as keys and performance for each run (a list of all runs. Each run contains a list of
            performances over epochs)
    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = get_rbp_shared_rocket_models()[0]
        >>> my_metrics = get_metrics(path=my_path)
        >>> list(my_metrics.keys())  # doctest: +NORMALIZE_WHITESPACE
        ['val_reduced3cleanedchildmindinstitute', 'val_cleanedchildmindinstitute', 'train_all',
         'test_cleanedchildmindinstitute', 'test_reduced3cleanedchildmindinstitute',
         'test_reduced1cleanedchildmindinstitute', 'val_reduced1cleanedchildmindinstitute']
        >>> list(my_metrics['val_cleanedchildmindinstitute'].keys())
        ['f1', 'precision', 'mcc', 'specificity', 'sensitivity', 'recall', 'accuracy', 'auc']
    """
    # Extract only folders which starts with 'Run_'
    run_folders = os.listdir(path)
    run_folders = [run_folder for run_folder in run_folders if run_folder[:4] == "Run_"]

    # Initialise metrics
    main_metrics: Dict[str, Dict[str, List[List[float]]]] = {}

    # Loop through all runs
    for run_folder in run_folders:
        # Get the name of all metric histories available. Extract only the elements which ends with '_history.pkl'
        history_names = os.listdir(path=f"{path}/{run_folder}")
        history_names = [history_name[:-12] for history_name in history_names if history_name[-12:] == "_history.pkl"]

        # Loop through all of them
        for history_name in history_names:
            # Extract the metrics of the current run and history
            metrics_dir = f"{path}/{run_folder}/{history_name}_history.pkl"
            with open(metrics_dir, "rb") as file:
                metrics = pickle.load(file)

            # Append it to main_metrics. Since main_metrics is not well initialised, it needs to be passed as None the
            # first time.
            try:
                main_metrics[history_name] = append_values(append_dict=metrics, main_dict=main_metrics[history_name])
            except KeyError:
                # main_metrics[history_name] raises KeyError
                main_metrics[history_name] = append_values(append_dict=metrics)

    return main_metrics


def get_metrics_single_run(path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Get all the metrics in a path (e.g. train_history.pkl and val_history.pkl). All elements in the folder which has
    a name ending with '_history.pkl' is interpreted as a metric Dict to load
    Args:
        path: Path to folder
    Returns:
        A Dictionary with all metrics available. Keys are e.g. 'train', 'val'. The values are dictionaries with metric
            name (e.g. 'accuracy') as keys and performance for the single run
    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = f"{get_rbp_shared_rocket_models()[0]}/Run_0"
        >>> my_metrics = get_metrics_single_run(path=my_path)
        >>> list(my_metrics.keys())  # doctest: +NORMALIZE_WHITESPACE
        ['val_reduced3cleanedchildmindinstitute', 'val_cleanedchildmindinstitute', 'train_all',
         'test_cleanedchildmindinstitute', 'test_reduced3cleanedchildmindinstitute',
         'test_reduced1cleanedchildmindinstitute', 'val_reduced1cleanedchildmindinstitute']
        >>> list(my_metrics['val_cleanedchildmindinstitute'].keys())
        ['f1', 'precision', 'mcc', 'specificity', 'sensitivity', 'recall', 'accuracy', 'auc']
        >>> type(my_metrics["val_cleanedchildmindinstitute"]["auc"])
        <class 'list'>
    """
    # Initialise metrics
    main_metrics: Dict[str, Dict[str, List[float]]] = {}

    # Get the name of all metric histories available. Extract only the elements which ends with '_history.pkl'
    history_names = os.listdir(path=path)
    history_names = [history_name[:-12] for history_name in history_names if history_name[-12:] == "_history.pkl"]

    # Loop through all of them
    for history_name in history_names:
        # Extract the metrics of the current run and history
        metrics_dir = os.path.join(path, f"{history_name}_history.pkl")
        with open(metrics_dir, "rb") as file:
            metrics = pickle.load(file)

        # Store in to main_metrics
        main_metrics[history_name] = metrics

    return main_metrics
