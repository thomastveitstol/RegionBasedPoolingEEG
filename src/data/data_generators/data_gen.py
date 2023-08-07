"""
Data Generators.
"""
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy
import torch
from torch.utils.data import Dataset

from src.data.datasets.data_base import EEGDataset, channel_names_to_indices


# ----------------------------------
# Data generators for models supporting
# pre-computing
# ----------------------------------
class PrecomputingDataGenerator(Dataset):
    """
    Data generator for models supporting pre-computing
    """

    def __init__(self, subjects: List[str], dataset: EEGDataset, target: str, time_series_start: int = 0,
                 seq_length: Optional[int] = None, device: Optional[torch.device] = None):
        super().__init__()

        # -----------------------------
        # Load input and targets
        # -----------------------------
        self._x = torch.tensor(dataset.load_eeg_data(subjects=subjects, time_series_start=time_series_start,
                                                     truncate=seq_length), dtype=torch.float32)
        self._y = torch.tensor(dataset.load_targets(subjects=subjects, target=target), dtype=torch.float32)

        # Initialise precomputed features
        self._pre_computed_features: Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]] = {}

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]]]:
        features = {}
        for key, value in self._pre_computed_features.items():
            features[key] = tuple([tuple([group[item].to(self._device) for group in path]) for path in value])
        return (self._x[item], torch.unsqueeze(self._y[item], dim=-1)), features

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def pre_computed_features(self) -> Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Get pre-computed features. Note that since python Dicts are mutable, the dictionary can be changed outside
        this class, despite not having a setter-method"""
        return self._pre_computed_features


class NewPrecomputingDataGenerator(Dataset):
    """
    Data generator for models supporting pre-computing
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: Optional[torch.device] = None):
        super().__init__()

        # -----------------------------
        # Load input and targets
        # -----------------------------
        self._x = x
        self._y = y

        # Initialise precomputed features
        self._pre_computed_features: Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]] = {}

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]]]:
        features = {}
        for key, value in self._pre_computed_features.items():
            features[key] = tuple([tuple([group[item].to(self._device) for group in path]) for path in value])
        return self._x[item], torch.unsqueeze(self._y[item], dim=-1), features

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def pre_computed_features(self) -> Dict[str, Tuple[Tuple[torch.Tensor, ...], ...]]:
        """Get pre-computed features. Note that since python Dicts are mutable, the dictionary can be changed outside
        this class, despite not having a setter-method"""
        return self._pre_computed_features


class StaticChannelPrecomputingDataGenerator(Dataset):
    """
    Data generator for models which support pre-computing, but the pre-computed features are computed per channel. This
    can e.g. be used when all channel splits share ROCKET feature extractor.
    """

    def __init__(self, subjects: List[str], dataset: EEGDataset, target: str, time_series_start: int = 0,
                 seq_length: Optional[int] = None):
        # -----------------------------
        # Load input and targets
        # -----------------------------
        self._x = torch.tensor(dataset.load_eeg_data(subjects=subjects, time_series_start=time_series_start,
                                                     truncate=seq_length), dtype=torch.float32)
        self._y = torch.tensor(dataset.load_targets(subjects=subjects, target=target), dtype=torch.float32)

        # Define precomputed features
        self._precomputed_features = None

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a tuple as (x, y, precomputed_features)"""
        return self._x[item], torch.unsqueeze(self._y[item], dim=-1), self._precomputed_features[item]

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def pre_computed_features(self) -> torch.Tensor:
        """Get pre-computed features"""
        return self._pre_computed_features

    @pre_computed_features.setter
    def pre_computed_features(self, value: torch.Tensor) -> None:
        """Setter method, to add the precomputed features to the Self object"""
        self._precomputed_features = value


class NewStaticChannelPrecomputingDataGenerator(Dataset):
    """
    Same as StaticChannelPrecomputingDataGenerator, but arguments to __init__ is different
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        # -----------------------------
        # Load input and targets
        # -----------------------------
        self._x = x
        self._y = y

        # Define precomputed features
        self._precomputed_features = None

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a tuple as (x, y, precomputed_features)"""
        precomputed_features = torch.tensor([0.]) if self._precomputed_features is None \
            else self._precomputed_features[item]  # not an optimal solution
        return self._x[item], torch.unsqueeze(self._y[item], dim=-1), precomputed_features

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def pre_computed_features(self) -> torch.Tensor:
        """Get pre-computed features"""
        return self._pre_computed_features

    @pre_computed_features.setter
    def pre_computed_features(self, value: torch.Tensor) -> None:
        """Setter method, to add the precomputed features to the Self object"""
        self._precomputed_features = value


# ----------------------------------
# Data generators for models not
# supporting pre-computing
# ----------------------------------
class PlainDataGenerator(Dataset):
    """
    Plain data generator
    """

    def __init__(self, subjects: List[str], dataset: EEGDataset, target: str, time_series_start: int = 0,
                 seq_length: Optional[int] = None, illegal_channels: Optional[Union[List[str], Tuple[str, ...]]] = None,
                 channel_name_to_index: Optional[Dict[str, int]] = None):
        super().__init__()

        # -----------------------------
        # Load input and targets
        # -----------------------------
        x = dataset.load_eeg_data(subjects=subjects, time_series_start=time_series_start, truncate=seq_length)
        y = dataset.load_targets(subjects=subjects, target=target)

        # Maybe remove illegal channels
        if illegal_channels is not None:
            illegal_indices = channel_names_to_indices(channel_names=illegal_channels,
                                                       channel_name_to_index=channel_name_to_index)
            x = numpy.delete(x, obj=illegal_indices, axis=1)

        # Store EEG and targets
        self._x = torch.tensor(x, dtype=torch.float32)
        self._y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x[item], torch.unsqueeze(self._y[item], dim=-1)


class ZeroFillDataGenerator(Dataset):
    """
    Data generator for zero-filling missing channels.

    todo: This is not memory efficient, and has room for improvement. It was written like this to get a quick-fix before
        a PhD course, and I wanted to run the experiments during the time I was taking the course.
    """

    def __init__(self, subjects: List[str], dataset: EEGDataset, target: str, time_series_start: int = 0,
                 seq_length: Optional[int] = None):
        super().__init__()

        # -----------------------------
        # Load input and targets
        # -----------------------------
        x = dataset.load_eeg_data(subjects=subjects, time_series_start=time_series_start, truncate=seq_length)

        self._x = torch.tensor(x, dtype=torch.float32)
        self._x_zero_filled: Dict[str, torch.Tensor] = dict()

        self._y = torch.tensor(dataset.load_targets(subjects=subjects, target=target), dtype=torch.float32)

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[Tuple[torch.tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        return (self._x[item], self._y[item]), \
            {channel_name: zero_filled[item] for channel_name, zero_filled in self._x_zero_filled.items()}

    def clear_memory(self) -> None:
        """Set all attributes to None"""
        self._x = None
        self._x_zero_filled = None
        self._y = None

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def x_interpolated(self) -> Dict[str, torch.Tensor]:
        """Get the zero-filled data. todo: not the best name of the property, but it works as a quick fix"""
        return self._x_zero_filled


class InterpolatingDataGenerator(Dataset):
    """
    Data generator for interpolating missing channels.

    Using MNE for interpolation:
    https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
    """

    def __init__(self, subjects: List[str], dataset: EEGDataset, target: str, time_series_start: int = 0,
                 seq_length: Optional[int] = None):
        super().__init__()

        # -----------------------------
        # Load input and targets
        # -----------------------------
        x = dataset.load_eeg_data(subjects=subjects, time_series_start=time_series_start, truncate=seq_length)

        self._x = torch.tensor(x, dtype=torch.float32)
        self._x_interpolated: Dict[str, torch.Tensor] = {}

        self._y = torch.tensor(dataset.load_targets(subjects=subjects, target=target), dtype=torch.float32)

    def __len__(self) -> int:
        return self._x.size()[0]

    def __getitem__(self, item) -> Tuple[Tuple[torch.tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        return (self._x[item], self._y[item]), \
            {channel_name: interpolated[item] for channel_name, interpolated in self._x_interpolated.items()}

    def clear_memory(self) -> None:
        """Set all attributes to None"""
        self._x = None
        self._x_interpolated = None
        self._y = None

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def x_interpolated(self) -> Dict[str, torch.Tensor]:
        return self._x_interpolated


# ----------------------------------
# Functions for removing channels for models
# not able to handle a varied number of channels.
# ----------------------------------
def zero_fill_input(x: torch.Tensor, channel_indices: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
    """
    Zero-fill input channels
    Args:
        x: a torch tensor containing EEG data with shape=(batch, channels, time steps).
        channel_indices: Channel indices to zero-fill. The indices should correspond to the excluded/removed channels

    Returns:
        a torch tensor of same shape as input 'x', where the time-steps of the channels in 'channel_indices' have been
        zero-filled for all elements in the batch
    Examples:
        >>> _ = torch.manual_seed(95)
        >>> my_x = torch.rand(size=(10, 4, 9))  # shape=(batch, channels, time steps)
        >>> my_outs = zero_fill_input(x=my_x, channel_indices=(2, 1))
        >>> my_outs[0]  # second and first channels have been zero-filled for zeroth subject in batch
        tensor([[0.5737, 0.1411, 0.8161, 0.8392, 0.9081, 0.9018, 0.0450, 0.4321, 0.4437],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.4486, 0.0800, 0.5089, 0.2561, 0.6096, 0.5586, 0.8597, 0.2385, 0.1870]])
        >>> my_outs[7]  # second and first channels have been zero-filled for seventh subject in batch
        tensor([[0.1082, 0.0694, 0.6754, 0.4706, 0.4181, 0.6504, 0.5656, 0.2408, 0.0945],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5995, 0.5871, 0.4928, 0.3290, 0.2166, 0.3112, 0.3865, 0.9649, 0.3443]])
        >>> my_x[0]  # input tensor has not been changed
        tensor([[0.5737, 0.1411, 0.8161, 0.8392, 0.9081, 0.9018, 0.0450, 0.4321, 0.4437],
                [0.2886, 0.5823, 0.0332, 0.4048, 0.3298, 0.4143, 0.1580, 0.5558, 0.6716],
                [0.5334, 0.4552, 0.1420, 0.5349, 0.3738, 0.8925, 0.5314, 0.6026, 0.3948],
                [0.4486, 0.0800, 0.5089, 0.2561, 0.6096, 0.5586, 0.8597, 0.2385, 0.1870]])
        >>> # Raises IndexError if there is a channel index out of bounds
        >>> zero_fill_input(x=torch.rand(size=(3, 6, 100)), channel_indices=(3, 6))
        Traceback (most recent call last):
        ...
        IndexError: index 6 is out of bounds for dimension 0 with size 6
    """
    # Need to clone the input first, to avoid changing the input object directly
    zero_filled = torch.clone(x)

    # Set channels to zero
    zero_filled[:, channel_indices] *= 0
    return zero_filled


def batch_interpolate_bad_channels(x: numpy.ndarray,
                                   all_channel_names: List[str],
                                   bad_channels: List[str],
                                   sampling_freq: Union[int, float],
                                   method: str = "spline") -> numpy.ndarray:
    """
    Using MNE to interpolate a selection of channels, for an entire batch of EEG data. No automatic detection of bad
    channels is done, and the channels passed in 'bad_channels' are interpolated for all EEGs in the batch
    Args:
        x: EEG data, a numpy array with shape=(batch, channels, time_steps)
        all_channel_names: see interpolate_bad_channels
        bad_channels: see interpolate_bad_channels
        sampling_freq: see interpolate_bad_channels
        method: see interpolate_bad_channels

    Returns:
        a numpy array with same shape as input ('x'), but the selected channels are interpolated
    Examples:
        >>> from src.data.datasets.data_base import channel_names_to_indices
        >>> from src.data.datasets.cleaned_child_data import CleanedChildData
        >>> my_dataset = CleanedChildData()
        >>> subject = my_dataset.get_subject_ids()[:4]
        >>> my_x = my_dataset.load_eeg_data(subjects=subject)  # shape = (batch, channels, time_steps)
        >>> illegal_channels = ['E2', 'E5', 'E50']
        >>> my_x[:, channel_names_to_indices(channel_names=illegal_channels,
        ...                            channel_name_to_index=my_dataset.channel_system.channel_name_to_index())] *= 0
        >>> my_channels = list(my_dataset.channel_system.channel_name_to_index().keys())
        >>> # Do interpolation
        >>> my_interpolated = batch_interpolate_bad_channels(x=my_x, all_channel_names=my_channels,
        ...                                                  bad_channels=illegal_channels, sampling_freq=500)
        >>> # Check output shape
        >>> my_interpolated.shape == my_x.shape
        True
        >>> # Check if the input is unchanged
        >>> my_time_steps = my_x.shape[-1]
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_x[:, 1])  # E2
        True
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_x[:, 4])  # E5
        True
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_x[:, 49])  # E50
        True
        >>> # Check if the interpolation is still zero-filled
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_interpolated[:, 1])  # E2
        False
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_interpolated[:, 4])  # E5
        False
        >>> numpy.allclose(numpy.zeros(shape=(4, my_time_steps)), my_interpolated[:, 49])  # E50
        False
        >>> my_interpolated = batch_interpolate_bad_channels(x=my_x, all_channel_names=my_channels,
        ...                                                  bad_channels=[], sampling_freq=500)
        >>> numpy.allclose(my_interpolated, my_x)  # No difference if no bad channels
        True
        >>> # ValueError if wrong input dimensions
        >>> batch_interpolate_bad_channels(x=my_x[0], all_channel_names=my_channels,
        ...                                bad_channels=illegal_channels, sampling_freq=500)
        Traceback (most recent call last):
        ...
        ValueError: Expected input 'x' to have three dimensions (batch, channels, time_steps), but found 2
    """
    # Input check
    if x.ndim != 3:
        raise ValueError(f"Expected input 'x' to have three dimensions (batch, channels, time_steps), but found "
                         f"{x.ndim}")

    # Initialise interpolated EEG as zeros
    interpolated_eeg = numpy.zeros(shape=x.shape)

    # Loop through all subject in the batch
    for i, raw_subject_eeg in enumerate(x):
        # Interpolate and add it to the interpolated data
        interpolated_eeg[i] = interpolate_bad_channels(x=raw_subject_eeg, all_channel_names=all_channel_names,
                                                       bad_channels=bad_channels, sampling_freq=sampling_freq,
                                                       method=method).copy()

    return interpolated_eeg


def interpolate_bad_channels(x: numpy.ndarray,
                             all_channel_names: Union[List[str], Tuple[str, ...]],
                             bad_channels: Union[List[str], Tuple[str, ...]],
                             sampling_freq: Union[int, float],
                             method: str = "spline") -> numpy.ndarray:
    """
    Using MNE to interpolate bad channels of a single subject EEG. Currently only compatible with GSN-HydroCel-129
    Args:
        x: a numpy array with shape=(channels, time_steps). That is, no batch dimension
        all_channel_names: All channel names (including the bad channels). The i-th element of this list must correspond
            to the i-th channel in the EEG input 'x'
        bad_channels: A list pr tuple of channel names to be interpolated
        sampling_freq: Sampling frequency. Needed for creating info-object of MNE
        method: Interpolation method to use. Defaults to "spline", as this is the default in the interpolate_bads()
            method of MNE

    Returns:
        a numpy array with same shape as input ('x'), but the bad channels are interpolated
    Examples:
        >>> from src.data.datasets.data_base import channel_names_to_indices
        >>> from src.data.datasets.cleaned_child_data import CleanedChildData
        >>> my_dataset = CleanedChildData()
        >>> subject = my_dataset.get_subject_ids()[0]
        >>> my_x = my_dataset.load_eeg_data(subjects=[subject])[0]  # shape = (channels, time_steps)
        >>> illegal_channels = ['E2', 'E5', 'E50']
        >>> my_x[list(channel_names_to_indices(channel_names=illegal_channels,
        ...                            channel_name_to_index=my_dataset.channel_system.channel_name_to_index()))] *= 0
        >>> my_channels = list(my_dataset.channel_system.channel_name_to_index().keys())
        >>> # Do interpolation
        >>> my_interpolated = interpolate_bad_channels(x=my_x, all_channel_names=my_channels,
        ...                                            bad_channels=illegal_channels, sampling_freq=500)
        >>> # Verify that the shapes are the same
        >>> my_x.shape == my_interpolated.shape
        True
        >>> # Verify that the bad channels of x are still zeros
        >>> my_time_steps = my_x.shape[-1]
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_x[1])  # E2
        True
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_x[4])  # E5
        True
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_x[49])  # E50
        True
        >>> # Verify that the bad channels of interpolated are non-zero
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_interpolated[1])  # E2
        False
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_interpolated[4])  # E5
        False
        >>> numpy.allclose(numpy.zeros(shape=my_time_steps), my_interpolated[49])  # E50
        False
        >>> # ValueError if wrong input dimensions
        >>> interpolate_bad_channels(x=numpy.expand_dims(my_x, axis=0), all_channel_names=my_channels,
        ...                          bad_channels=illegal_channels, sampling_freq=500)
        Traceback (most recent call last):
        ...
        ValueError: Expected input 'x' to have two dimensions (channels, time_steps), but found 3
    """
    # Input check
    if x.ndim != 2:
        raise ValueError(f"Expected input 'x' to have two dimensions (channels, time_steps), but found {x.ndim}")

    # ----------------
    # To MNE RawArray object
    # ----------------
    # Create info object and set bad channels
    info = mne.create_info(ch_names=all_channel_names, sfreq=sampling_freq, ch_types="eeg")
    info["bads"] = bad_channels

    # Set montage. This is important for getting channel positions and needed for interpolation
    info.set_montage("GSN-HydroCel-129")  # 129 to include Cz

    # Create mne.io.RawArray object
    eeg = mne.io.RawArray(data=x.copy(), info=info, verbose=False)

    # ----------------
    # Interpolate
    # ----------------
    eeg.interpolate_bads(method={"eeg": method}, verbose=False)

    return eeg.get_data()
