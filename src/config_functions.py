import configparser
from typing import Any, Dict, List, Optional, Tuple

from src.data.datasets.getters import get_channel_system, get_dataset


def get_args_str_and_arg_type(string: str) -> Tuple[str, str]:
    """
    Get the arg_str and the arg_type of a string, as often given from a config parser
    Args:
        string: String containing type and argument

    Returns: String argument and type as strings

    Examples:
        >>> get_args_str_and_arg_type("int: 4")
        ('4', 'int')
    """
    args = string.split(sep=":")

    assert len(args) == 2, "Splitting into arg_str and arg_type is only implemented with the presence of a single " \
                           "colon only"
    return args[1].strip(), args[0].strip()


def to_dict(config_section: configparser.SectionProxy) -> Dict[str, Any]:
    """
    Converts a config section to a dict
    Args:
        config_section: The config section to convert into a dictionary
    Returns:
        A dictionary with keys as in the config section and values as specified in the config sections (with correct
        types)
    """
    dictionary = {}
    for key, value in config_section.items():
        args = value.split(sep=":")
        if len(args) == 3:
            if args[0].strip().lower() == "list":
                dictionary[key] = str_to_list(arg_str=args[2], arg_type=args[1])
            elif args[0].strip().lower() == "tuple":
                dictionary[key] = str_to_list(arg_str=args[2], arg_type=args[1])
            else:
                raise ValueError(f"Unexpected: args: {args}")
        elif len(args) == 2:
            dictionary[key] = str_to_type(arg_str=args[-1], arg_type=args[0])
        else:
            raise ValueError(f"Unexpected: args: {args}")

    return dictionary


def str_to_type(arg_str: str, arg_type: str) -> Any:
    """
    Converts string to specified type
    :param arg_str: argument string
    :param arg_type: desired argument type
    :return: argument as the desired type
    Examples:
        >>> str_to_type("4.2", "float")
        4.2
        >>> str_to_type("7", "int")
        7
        >>> str_to_type("   hello darkness my old friend  ", "str")
        'hello darkness my old friend'
        >>> type(str_to_type("  True ", "bool"))
        <class 'bool'>
        >>> str_to_type("  False ", "   bool")
        False
        >>> str_to_type("Example", "ChannelSystem")
        --- Channel System ---
        Name: Example
        Number of channels: 200
        >>> str_to_type("ExampleData", "Dataset")
        -------------------
        --- EEG dataset ---
        <BLANKLINE>
        --- Channel System ---
        Name: Example
        Number of channels: 200
        --- Data ---
        Root directory: /not/a/real/dataset
        Maximum number of time steps allowed: 3000
        -------------------
        >>> str_to_type("true_false  ", "bool")
        Traceback (most recent call last):
        ...
        NameError: Boolean value true_false not recognised
        >>> str_to_type("5.5", "numpy.float64     ")
        Traceback (most recent call last):
        ...
        TypeError: Argument type 'numpy.float64' not understood
    """
    arg_type = arg_type.strip()
    arg_str = arg_str.strip()

    if arg_type == "int":
        return int(arg_str)
    elif arg_type == "float":
        return float(arg_str)
    elif arg_type == "str":
        return arg_str
    elif arg_type == "bool":
        if arg_str.lower() == "true":
            return True
        elif arg_str.lower() == "false":
            return False
        else:
            raise NameError(f"Boolean value {arg_str} not recognised")
    elif arg_type.lower() in ("channelsystem", "channel_system", "basechannelsystem", "base_channel_system"):
        return get_channel_system(name=arg_str)
    elif arg_type.lower() in ("dataset", "eegdataset", "eeg_dataset"):
        return get_dataset(name=arg_str)
    else:
        raise TypeError(f"Argument type '{arg_type}' not understood")


def str_to_optional_type(arg_str: str, arg_type: str) -> Optional[Any]:
    """
    Same as str_to_type, but output can be none
    Examples:
        >>> str_to_optional_type("4.2", "float")
        4.2
        >>> str_to_optional_type("7", "int")
        7
        >>> type(str_to_optional_type("none", "int"))
        <class 'NoneType'>
    """
    if arg_str in ["None", "none"]:
        outputs = None
    else:
        outputs = str_to_type(arg_str=arg_str, arg_type=arg_type)
    return outputs


def str_to_tuple(arg_str: str, arg_type: str) -> Tuple[Any, ...]:
    """
    Convert from string to list. Handy to use when loading values from config files
    Args:
        arg_str: The string to be converted to a list
        arg_type: The type of the elements in the list
    Returns: List with elements of type arg_type
    Examples:
        >>> my_list = "   (4, 3, 7, 9, 5) "
        >>> str_to_tuple(arg_str=my_list, arg_type="int")
        (4, 3, 7, 9, 5)
        >>> str_to_tuple(arg_str=my_list, arg_type="float")
        (4.0, 3.0, 7.0, 9.0, 5.0)
    """
    elements = []

    arg_list = arg_str.strip()[1:-1].split(sep=",")  # Remove brackets and split on comma
    for element in arg_list:
        elements.append(str_to_type(arg_str=element, arg_type=arg_type))
    return tuple(elements)


def str_to_list(arg_str: str, arg_type: str) -> List[Any]:
    """
    Convert from string to list. Handy to use when loading values from config files
    Args:
        arg_str: The string to be converted to a list
        arg_type: The type of the elements in the list

    Returns:
        List with elements of type arg_type

    Examples:
        >>> my_list = "   [4, 3, 7, 9, 5] "
        >>> str_to_list(arg_str=my_list, arg_type="int")
        [4, 3, 7, 9, 5]
        >>> str_to_list(arg_str=my_list, arg_type="float")
        [4.0, 3.0, 7.0, 9.0, 5.0]
    """
    elements = []

    arg_list = arg_str.strip()[1:-1].split(sep=",")  # Remove brackets and split on comma
    for element in arg_list:
        elements.append(str_to_type(arg_str=element, arg_type=arg_type))
    return elements


def str_to_optional_list(arg_str: str, arg_type: str) -> Optional[List[Any]]:
    """
    Same as str_to_list, but returns None if the arg_str is None or none

    Examples:
        >>> my_list = "None"
        >>> my_output = str_to_optional_list(arg_str=my_list, arg_type="int")
        >>> type(my_output)
        <class 'NoneType'>
        >>> my_list = "[4, 3, 7, 9, 5]"
        >>> str_to_optional_list(arg_str=my_list, arg_type="float")
        [4.0, 3.0, 7.0, 9.0, 5.0]
    """
    if arg_str in ["None", "none"]:
        outputs = None
    else:
        outputs = str_to_list(arg_str=arg_str, arg_type=arg_type)
    return outputs
