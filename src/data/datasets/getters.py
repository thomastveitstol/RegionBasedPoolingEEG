from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem, Reduced1CleanedChildChannelSystem, \
    Reduced3CleanedChildChannelSystem, CleanedChildData
from src.data.datasets.data_base import BaseChannelSystem, EEGDataset
from src.data.datasets.example_data import ExampleChannelSystem, ExampleReducedChannelSystem, ExampleData


def get_channel_system(name: str) -> BaseChannelSystem:
    """
    Get a channel system by its name
    Args:
        name: Name of the channel system to get

    Returns:
        A channel system object
    Examples:
        >>> get_channel_system("Example")
        --- Channel System ---
        Name: Example
        Number of channels: 200
        >>> get_channel_system("CleanedChildMindInstitute")
        --- Channel System ---
        Name: CleanedChildMindInstitute
        Number of channels: 129
        >>> _ = get_channel_system("ReducedExample")
        >>> _ = get_channel_system("NotAChannelSystem")
        Traceback (most recent call last):
        ...
        ValueError: The channel system NotAChannelSystem was not recognised
    """
    available_channel_systems = (ExampleChannelSystem(), ExampleReducedChannelSystem(), CleanedChildChannelSystem(),
                                 Reduced1CleanedChildChannelSystem(), Reduced3CleanedChildChannelSystem())

    for channel_system in available_channel_systems:
        if channel_system.name == name:
            return channel_system

    raise ValueError(f"The channel system {name} was not recognised")


def get_dataset(name: str) -> EEGDataset:
    """
    Get a dataset by its name
    Args:
        name: Class name of the dataset to get

    Returns:
        an EEG dataset
    Examples:
        >>> get_dataset("ExampleData")
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
        >>> get_dataset("CleanedChildData")
        -------------------
        --- EEG dataset ---
        <BLANKLINE>
        --- Channel System ---
        Name: CleanedChildMindInstitute
        Number of channels: 129
        --- Data ---
        Root directory: /home/thomas/ChildMindInstituteCleaned
        Maximum number of time steps allowed: 15000
        -------------------
        >>> _ = get_dataset("NotARealDataset")
        Traceback (most recent call last):
        ...
        ValueError: The dataset NotARealDataset was not recognised
    """
    available_datasets = (ExampleData(), CleanedChildData())
    for dataset in available_datasets:
        if type(dataset).__name__ == name:
            return dataset

    raise ValueError(f"The dataset {name} was not recognised")
