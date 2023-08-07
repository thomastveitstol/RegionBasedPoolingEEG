"""
Base classes for channel systems and EEGDatasets.
"""
import abc
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy
from matplotlib import pyplot

from src.utils import CartesianCoordinates, PolarCoordinates, to_cartesian


# -----------------------------------
# Base classes for Channel Systems
# -----------------------------------
# For channel systems in general
class BaseChannelSystem(abc.ABC):
    """
    Base class for Channel Systems
    """
    __slots__ = "_name", "_num_channels"

    def __init__(self, name: str, num_channels: int) -> None:
        self._name = name
        self._num_channels = num_channels

    # --------------------
    # Abstract methods
    # --------------------
    @staticmethod
    @abc.abstractmethod
    def get_electrode_positions() -> Dict[str, Union[PolarCoordinates, CartesianCoordinates]]:
        """
        Get a Dict containing the positions of the electrodes. The keys are channel names such as 'Cz', the values are
        the corresponding coordinates.
        """

    @staticmethod
    @abc.abstractmethod
    def channel_name_to_index() -> Dict[str, int]:
        """
        Return a Dictionary which maps channel name to index in the numpy arrays
        """

    # --------------------
    # Plotting
    # --------------------
    def plot_electrode_positions(self, annotate: bool = True) -> None:
        """
        Method for 3D plotting the electrode positions. This method does not call pyplot.show()
        Args:
            annotate: To show the channel names (True) or not (False)

        Returns: Nothing, it just plots the electrode positions

        """
        electrode_positions = self.get_electrode_positions()

        # Extract x, y, z values
        x = [to_cartesian(electrode_position).coordinates[0] for electrode_position in electrode_positions.values()]
        y = [to_cartesian(electrode_position).coordinates[1] for electrode_position in electrode_positions.values()]
        z = [to_cartesian(electrode_position).coordinates[2] for electrode_position in electrode_positions.values()]

        # Make new figure
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot
        ax.scatter(x, y, z)

        # Annotate the channels with channel names (if desired)
        if annotate:
            for x_pos, y_pos, z_pos, channel in zip(x, y, z, electrode_positions):
                ax.text(x=x_pos, y=y_pos, z=z_pos, s=channel)

    # --------------------
    # Properties
    # --------------------
    @property
    def name(self) -> str:
        """The name of the channel system"""
        return self._name

    @property
    def num_channels(self) -> int:
        """Get the number of channels"""
        return self._num_channels

    # --------------------
    # Printing
    # --------------------
    def __repr__(self) -> str:
        return f"--- Channel System ---\n" \
               f"Name: {self._name}\n" \
               f"Number of channels: {self._num_channels}"


# For reduced channel systems
class ReducedBaseChannelSystem(BaseChannelSystem, abc.ABC):
    """
    This class is inherited from when it is a low-resolution equivalent of an already implemented channel system (which
    inherits from BaseChannelSystem)
    """

    original_channel_system: BaseChannelSystem
    channel_names: Tuple[str, ...]  # All channel names must be in the original channel system as well

    def __init__(self, reduced_name: str, num_channels: int):
        # --------------------------
        # Class check:
        #
        # Ensure that original_channel_system is set
        # with correct type. If not set at all, an
        # AttributeError is raised automatically instead
        # --------------------------
        if not isinstance(self.original_channel_system, BaseChannelSystem):
            raise TypeError(f"Expected the Original Channel System to be of type BaseChannelSystem (or a subclass "
                            f"of it), but found {type(self.original_channel_system)}")
        super().__init__(name=f"{reduced_name}{self.original_channel_system.name}", num_channels=num_channels)

    def channel_name_to_index(self) -> Dict[str, int]:
        """Channel name to index should be the same as its original channel system object. This may be confusing, but
        as classes inheriting from ReducedBaseChannelSystem are used in combination with EEGDataset objects, this will
        lead to correct behaviour"""
        return self.original_channel_system.channel_name_to_index()


# -----------------------------------
# Base class for EEG Datasets
# -----------------------------------
class EEGDataset(abc.ABC):
    """
    Base class for all datasets to be used. Classes for datasets such as ChildMindInstitute Dataset (although other
    datasets are not used in the paper, I decided it would be smart to make it scalable from the beginning), should
    inherit from this class
    """

    __slots__ = "_root_dir", "_max_time_steps", "_channel_system"

    def __init__(self, root_dir: str, max_time_steps: int, channel_system: BaseChannelSystem) -> None:
        """
        Initialise
        Args:
            root_dir: Root directory. Commonly used in methods such as data loading
            max_time_steps: Maximum number of time steps (e.g. 30000 if the duration is 60s and sampling freq is 500Hz)
            channel_system: The channel system which is used (an object inheriting from BaseChannelSystem)
        """
        self._root_dir = root_dir
        self._max_time_steps = max_time_steps
        self._channel_system = channel_system

    # --------------------------------
    # Methods for subject related info
    # --------------------------------
    @staticmethod
    @abc.abstractmethod
    def get_subject_ids() -> List[str]:
        """Method for obtaining all IDs contained in the dataset"""

    @classmethod
    @abc.abstractmethod
    def get_subject_sex(cls) -> Dict[str, int]:
        """Get a dict containing the sex of all subjects. The keys are subject IDs and the values are 0 if male and 1 if
        female"""

    @classmethod
    @abc.abstractmethod
    def get_subject_age(cls) -> Dict[str, float]:
        """Get a dict containing the age of all subjects. The keys are subject IDs, the values are age"""

    # --------------------------------
    # Method for loading data
    # --------------------------------
    @abc.abstractmethod
    def load_eeg_data(self, subjects: List[str], time_series_start: int = 0,
                      truncate: Optional[int] = None) -> numpy.ndarray:
        """
        Method for loading in the EEG data of the given subjects. Output will be a numpy.ndarray with
        shape=(num_subject, num_channels, num_time_steps), with num_time_steps=truncate

        Args:
            subjects: List of subject IDs
            time_series_start: At which time point to start from. It is often convenient not to start at 0, as the first
                seconds are more likely to contain artifacts
            truncate: Number of time steps to load (starting from time_series_start). If None, all data available will
                be used
        """

    def check_inputs_load_eeg_data(self, time_series_start: int, truncate: Optional[int] = None) -> None:
        """
        Check if the number of planned time steps to load in the EEG data exceeds the maximum allowed
        Args:
            time_series_start: At what time step to start
            truncate: Length of the time series. If None, all available data from time_series_start will be used

        Returns:
            Nothing, this is just an input check
        """
        max_length = self.max_time_steps
        truncate = max_length - time_series_start if truncate is None else truncate
        if time_series_start + truncate > max_length:
            raise ValueError(
                f"The time series cannot exceed the end of the stored numpy arrays (length {max_length}), but starts "
                f"at {time_series_start} and ends at {time_series_start + truncate}.")

    @abc.abstractmethod
    def load_targets(self, target: str, subjects: List[str]) -> numpy.array:
        """Method for loading targets. Output will be a numpy.ndarray with shape=(num_subjects, Any), depending on what
        target is used"""

    def _sex_split_to_lists(self, subjects: List[str]) -> Tuple[List[str], List[str]]:
        """
        Splits the subjects into two lists, one list containing males only, the other containing females only. For every
        subject in the list which is not found in the dict returned by get_subject_sex, the subject ID will be printed
        as excluded
        Returns:
            A tuple of lists. The first element contains the subject IDs of all included males, the second element
            contains the subject IDs of all included females
        """
        # Get dictionary with sex subject ID as keys and sex as values
        subject_sex = self.get_subject_sex()

        # Loop through all participants and store male/female in different lists
        male_ids = []
        female_ids = []
        for subject in subjects:
            try:
                sex = subject_sex[subject]
                if sex == 0:
                    male_ids.append(subject)
                elif sex == 1:
                    female_ids.append(subject)
                else:
                    raise ValueError(f"Subject sex {subject_sex} not recognised")
            except KeyError:
                # This means that the sex of the subject is unknown
                print(f"Subject excluded: {subject}")

        return male_ids, female_ids

    @staticmethod
    def k_fold_split(subjects: Tuple[str, ...], num_folds: int, shuffle: bool) -> Tuple[Tuple[str, ...], ...]:
        """
        Method for splitting subjects into k folds

        Args:
            subjects: Subjects to split
            num_folds: Number of folds (k in k-fold cross validation)
            shuffle: To randomise the split (True) or not (False)

        Returns: Each element is a fold. Each fold contains a tuple of subjects (see Examples)

        Examples:
            >>> my_subjects = ("Magnus", "Checo", "Max", "Yuki", "Lewis", "Daniel", "Steiner", "Toto", "Lando", "Gene")
            >>> EEGDataset.k_fold_split(subjects=my_subjects, num_folds=3, shuffle=False)
            (('Magnus', 'Checo', 'Max', 'Yuki'), ('Lewis', 'Daniel', 'Steiner'), ('Toto', 'Lando', 'Gene'))
            >>> random.seed(2)
            >>> EEGDataset.k_fold_split(subjects=my_subjects, num_folds=3, shuffle=True)
            (('Daniel', 'Gene', 'Yuki', 'Lewis'), ('Steiner', 'Toto', 'Max'), ('Lando', 'Checo', 'Magnus'))
            >>> my_subjects = ("Magnus", "Checo", "Max", "Yuki", "Lewis", "Daniel", "Steiner", "Toto")
            >>> EEGDataset.k_fold_split(subjects=my_subjects, num_folds=3, shuffle=True)
            (('Toto', 'Steiner', 'Max'), ('Yuki', 'Checo', 'Daniel'), ('Lewis', 'Magnus'))
        """
        if shuffle:
            subjects = list(subjects)
            random.shuffle(subjects)
            subjects = tuple(subjects)

        return tuple(tuple(fold) for fold in numpy.array_split(subjects, num_folds))

    def k_fold_sex_split(self, subjects: Tuple[str, ...], num_folds: int, force_balanced: bool = False,
                         num_subjects: Optional[int] = None):
        """Method for splitting into k folds, with the additional property that all folds may be forced balanced by
        sex"""

        # If the data split should not be forced balanced, use the default method for splitting into folds
        if not force_balanced:
            subjects = subjects[:num_subjects] if num_subjects is not None else subjects
            self.k_fold_split(subjects=subjects, num_folds=num_folds, shuffle=True)

        # Loop through all subjects and store male/female in different lists. Also, exclude subjects which sex is
        # unknown
        male_ids, female_ids = self._sex_split_to_lists(subjects=list(subjects))

        # Total number of subjects is limited to the number of males and females
        if num_subjects is None:
            num_subjects = min(len(male_ids) * 2, len(female_ids) * 2)
        else:
            num_subjects = min(len(male_ids) * 2, len(female_ids) * 2, num_subjects)

        # Shuffle male and female IDs, separately
        numpy.random.shuffle(male_ids)
        numpy.random.shuffle(female_ids)

        # Force balanced, by down-sampling the class in abundance
        male_ids = male_ids[:num_subjects // 2]
        female_ids = female_ids[:num_subjects // 2]

        # Split into folds, separately
        male_folds = self.k_fold_split(subjects=tuple(male_ids), num_folds=num_folds, shuffle=True)
        female_folds = self.k_fold_split(subjects=tuple(female_ids), num_folds=num_folds, shuffle=True)

        # Merge the folds
        return _merge_folds(males=male_folds, females=female_folds)

    # --------------------
    # Properties
    # --------------------
    @property
    def name(self) -> str:
        """Get the name of the channel system"""
        return self._channel_system.name

    @property
    def root_dir(self) -> str:
        """Get the root directory of the dataset"""
        return self._root_dir

    @property
    def num_channels(self) -> int:
        """Get the number of channels"""
        return self._channel_system.num_channels

    @property
    def max_time_steps(self) -> int:
        """Get the maximum number of time steps"""
        return self._max_time_steps

    @property
    def channel_system(self) -> BaseChannelSystem:
        """Get the channel system"""
        return self._channel_system

    # --------------------
    # Printing
    # --------------------
    def __repr__(self) -> str:
        return f"-------------------\n" \
               f"--- EEG dataset ---\n" \
               f"\n" \
               f"{self._channel_system}\n" \
               f"--- Data ---\n" \
               f"Root directory: {self._root_dir}\n" \
               f"Maximum number of time steps allowed: {self._max_time_steps}\n" \
               f"-------------------"


# ----------------
# Functions
# -----------------
def _merge_folds(males: Tuple[Tuple[str, ...], ...], females: Tuple[Tuple[str, ...], ...]) \
        -> Tuple[Tuple[str, ...], ...]:
    """
    Merge male and female folds
    Args:
        males: Male subjects
        females: Females subjects

    Returns: A merging of the male and female folds
    Examples:
        >>> my_male_folds = (("m1", "m2", "m3"), ("m4", "m5", "m6"), ("m7", "m8"))
        >>> my_female_folds = (("f1", "f2", "f3"), ("f4", "f5", "f6"), ("f7", "f8"))
        >>> _merge_folds(males=my_male_folds, females=my_female_folds)
        (('m1', 'm2', 'm3', 'f1', 'f2', 'f3'), ('m4', 'm5', 'm6', 'f4', 'f5', 'f6'), ('m7', 'm8', 'f7', 'f8'))
        >>> my_male_folds = (("m1", "m2", "m3"), ("m4", "m5", "m6"), ("m7",))
        >>> my_female_folds = (("f1", "f3"), ("f4",), ("f7", "f8"))
        >>> _merge_folds(males=my_male_folds, females=my_female_folds)
        (('m1', 'm2', 'm3', 'f1', 'f3'), ('m4', 'm5', 'm6', 'f4'), ('m7', 'f7', 'f8'))
    """
    return tuple(m + f for m, f in zip(males, females))


def channel_names_to_indices(channel_names: Union[Tuple[str, ...], List[str]],
                             channel_name_to_index: Dict[str, int]) -> Tuple[int, ...]:
    """
    Same as channel_name_to_index, but now you can pass in a list/tuple of channel names
    Args:
        channel_names: Channel names to be mapped to indices
        channel_name_to_index: Object calculated from channel_name_to_index.

    Returns: The indices of the input channel names
    Examples:
        >>> my_relevant_channel_names = ["Cz", "POO10h", "FFT7h"]
        >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6}
        >>> channel_names_to_indices(channel_names=my_relevant_channel_names,
        ...                          channel_name_to_index=my_channel_name_to_index)
        (0, 4, 6)
        >>> channel_names_to_indices(channel_names=tuple(my_relevant_channel_names),
        ...                          channel_name_to_index=my_channel_name_to_index)
        (0, 4, 6)
    """
    return tuple(channel_name_to_index[channel_name] for channel_name in channel_names)


def smallest_channel_system(channel_systems: Union[Tuple[BaseChannelSystem, ...], List[BaseChannelSystem]]) \
        -> BaseChannelSystem:
    """
    Get the channel system which contains the smallest number of channels

    (unittests in test folder)
    Args:
        channel_systems: A list of channel systems

    Returns:
        The channel system which contains the smallest number of channels in the list
    """
    # Find the index
    idx = numpy.argmin([ch_system.num_channels for ch_system in channel_systems])

    # return the smallest channel system
    return channel_systems[idx]


def get_illegal_channels(main_channel_system: BaseChannelSystem,
                         reduced_channel_systems: Tuple[BaseChannelSystem, ...]) -> Dict[str, List[str]]:
    """
    Get the channel names of all illegal channels in a list of reduced channel systems

    (unit tests in test folder)
    Args:
        main_channel_system: The main channel system
        reduced_channel_systems: The reduced channel systems

    Returns:
        A Dict with keys as channel system names and values as illegal channel names
    """
    # Get all possible channel names from the main channel system
    all_channel_names = main_channel_system.channel_name_to_index().keys()

    # Set the illegal channel names of all reduced channel systems
    illegal_channels = {}
    for channel_system in reduced_channel_systems:
        # Get the legal channels of the current channel system
        legal_channels = channel_system.get_electrode_positions().keys()

        # Get the illegal channels of the current channel system
        illegal_channels[channel_system.name] = [channel for channel in all_channel_names
                                                 if channel not in legal_channels]

    return illegal_channels
