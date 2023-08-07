from typing import Dict, List, Optional, Union
import random

import numpy

from src.utils import CartesianCoordinates, PolarCoordinates, to_cartesian
from src.data.datasets.data_base import BaseChannelSystem, ReducedBaseChannelSystem, EEGDataset


# ------------------------------
# Dummy/examples channel systems
# ------------------------------
# Main channel system
class ExampleChannelSystem(BaseChannelSystem):

    def __init__(self):
        """
        Initialise
        Examples:
            >>> ExampleChannelSystem()
            --- Channel System ---
            Name: Example
            Number of channels: 200
        """
        super().__init__(name="Example", num_channels=200)

    @staticmethod
    def get_electrode_positions() -> Dict[str, CartesianCoordinates]:
        """
        Examples:
            >>> numpy.random.seed(2)
            >>> ExampleChannelSystem.get_electrode_positions()["Ch6"]
            CartesianCoordinates(coordinates=(-0.19597295915298235, 0.002730253098426223, 0.980605499168163))
        """
        # Make reproducible
        numpy.random.seed(3)

        channel_names = [f"Ch{i}" for i in range(200)]

        rho_positions = [1]*200
        theta_positions = numpy.random.uniform(0., numpy.pi/2, size=200)
        phi_positions = numpy.random.uniform(0., 2*numpy.pi, size=200)

        return {channel_name: to_cartesian(PolarCoordinates((rho, theta, phi)))
                for channel_name, rho, theta, phi in zip(channel_names,  rho_positions, theta_positions, phi_positions)}

    @staticmethod
    def channel_name_to_index() -> Dict[str, int]:
        """
        Examples:
            >>> ExampleChannelSystem.channel_name_to_index()["Ch8"]
            8
            >>> list(ExampleChannelSystem.channel_name_to_index().keys())[:11]
            ['Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10']
        """
        channel_names = [f"Ch{i}" for i in range(200)]
        return {channel_name: i for i, channel_name in enumerate(channel_names)}


# Reduced channel system
class ExampleReducedChannelSystem(ReducedBaseChannelSystem):

    original_channel_system = ExampleChannelSystem()

    def __init__(self) -> None:
        """
        Initialise
        Examples:
            >>> my_channel_system = ExampleReducedChannelSystem()
            >>> print(my_channel_system)
            --- Channel System ---
            Name: ReducedExample
            Number of channels: 100
        """
        super().__init__(reduced_name="Reduced", num_channels=self.original_channel_system.num_channels // 2)

    @classmethod
    def get_electrode_positions(cls) -> Dict[str, Union[PolarCoordinates, CartesianCoordinates]]:
        """
        Get electrode positions
        Returns:
            Electrode positions
        Examples:
            >>> my_channel_system = ExampleReducedChannelSystem()
            >>> list(my_channel_system.get_electrode_positions().keys())[:11]
            ['Ch0', 'Ch2', 'Ch4', 'Ch6', 'Ch8', 'Ch10', 'Ch12', 'Ch14', 'Ch16', 'Ch18', 'Ch20']
            >>> len(my_channel_system.get_electrode_positions())
            100
            >>> len(ExampleReducedChannelSystem.get_electrode_positions())  # Perks of class methods
            100
        """
        # Get all electrode positions (of the original channel system)
        all_electrode_positions = cls.original_channel_system.get_electrode_positions()

        # Keep every second channel
        return {ch_name: position for i, (ch_name, position) in enumerate(all_electrode_positions.items())
                if numpy.mod(i, 2) == 0}


# ------------------------------
# Dummy/examples EEG dataset
# ------------------------------
class ExampleData(EEGDataset):
    """
    An Example dataset
    """

    def __init__(self, root_dir="/not/a/real/dataset"):
        """
        Initialise
        Args:
            root_dir: Root directory
        Examples:
            >>> ExampleData()  # doctest: +NORMALIZE_WHITESPACE
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
        """
        super().__init__(root_dir=root_dir, max_time_steps=3000, channel_system=ExampleChannelSystem())

    @staticmethod
    def get_subject_ids() -> List[str]:
        """
        Get subject IDs
        Returns:
            List of all subject IDs
        Examples:
            >>> my_dataset = ExampleData()
            >>> my_dataset.get_subject_ids(), len(my_dataset.get_subject_ids())  # doctest: +NORMALIZE_WHITESPACE
            (['PDUFHPNPU', 'P8NE8ER9', 'OEEKPOPRRK', 'BHERIUHIUFRIJ', 'OIIUFIGOGD', 'ICUYNUIYNPN', 'OIHIHUYGO',
              'KHGCUYGBIUCUYB', 'UIYNCOIUERIOPCU', 'UYNVCNNNNHFHK'], 10)
        """
        return ["PDUFHPNPU", "P8NE8ER9", "OEEKPOPRRK", "BHERIUHIUFRIJ", "OIIUFIGOGD", "ICUYNUIYNPN", "OIHIHUYGO",
                "KHGCUYGBIUCUYB", "UIYNCOIUERIOPCU", "UYNVCNNNNHFHK"]

    @classmethod
    def get_subject_age(cls) -> Dict[str, float]:
        """
        Examples:
            >>> ExampleData.get_subject_age()  # doctest: +NORMALIZE_WHITESPACE
            {'PDUFHPNPU': 51.368619621413494, 'P8NE8ER9': 64.90071274515702, 'OEEKPOPRRK': 29.01780754651321,
             'BHERIUHIUFRIJ': 47.93117404699902, 'OIIUFIGOGD': 80.7934380738983, 'ICUYNUIYNPN': 81.08120564827567,
             'OIHIHUYGO': 14.800336699889918, 'KHGCUYGBIUCUYB': 21.822887519884063,
             'UIYNCOIUERIOPCU': 8.426179483871369, 'UYNVCNNNNHFHK': 41.90964655395474}

        """
        # Make reproducible
        numpy.random.seed(3)

        # Get all IDs
        subject_ids = cls.get_subject_ids()

        # Set age between 4 and 90
        return {subject_id: numpy.random.uniform(low=4, high=90) for subject_id in subject_ids}

    @classmethod
    def get_subject_sex(cls) -> Dict[str, int]:
        """
        Examples:
            >>> ExampleData.get_subject_sex()  # doctest: +NORMALIZE_WHITESPACE
            {'PDUFHPNPU': 0, 'P8NE8ER9': 0, 'OEEKPOPRRK': 1, 'BHERIUHIUFRIJ': 1, 'OIIUFIGOGD': 0, 'ICUYNUIYNPN': 0,
             'OIHIHUYGO': 1, 'KHGCUYGBIUCUYB': 1, 'UIYNCOIUERIOPCU': 0, 'UYNVCNNNNHFHK': 0}
        """
        # Make reproducible
        random.seed(3)

        # Get all subject IDs
        subject_ids = cls.get_subject_ids()

        # Sex generated randomly
        return {subject_id: random.randint(0, 1) for subject_id in subject_ids}

    # --------------------------------
    # Methods for loading data
    # --------------------------------
    def load_eeg_data(self, subjects: List[str],  time_series_start: int = 0,
                      truncate: Optional[int] = None) -> numpy.ndarray:
        """
        Method for loading in the EEG data of the given subjects
        Args:
            subjects: Subjects to load (list of IDs)
            time_series_start: At what time step to start. It is usually a good idea to not use the first seconds, as
                they often contain more artifacts
            truncate: Length of the time series. If None, all available data from time_series_start will be used

        Returns:
            A numpy array containing the EEG data with shape=(num_subjects, num_channels, num_time_steps). The i-th row
            of the numpy array corresponds to the i-th subject.
        Examples:
            >>> my_data = ExampleData()
            >>> my_data.load_eeg_data(subjects=["PDUFHPNPU", "P8NE8ER9", "OEEKPOPRRK"], time_series_start=200,
            ...                       truncate=300).shape
            (3, 200, 300)
            >>> my_data.load_eeg_data(subjects=["PDUFHPNPU", "P8NE8ER9", "OEEKPOPRRK"],
            ...                       time_series_start=25000, truncate=6000)  # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
            ...
            ValueError: The time series cannot exceed the end of the stored numpy arrays (length 3000), but starts at
                25000 and ends at 31000.
        """
        # -------------------------
        # Check inputs
        # -------------------------
        max_length = self.max_time_steps
        truncate = max_length - time_series_start if truncate is None else truncate
        if time_series_start + truncate > max_length:
            raise ValueError(
                f"The time series cannot exceed the end of the stored numpy arrays (length {max_length}), but starts "
                f"at {time_series_start} and ends at {time_series_start + truncate}.")

        # Return dummy data
        numpy.random.seed(95)
        return numpy.random.uniform(low=-1, high=1, size=(len(subjects), self.num_channels, truncate))

    def load_targets(self, target: str, subjects: List[str]) -> numpy.ndarray:
        """
        Method for loading targets

        (target methods tested separately)
        Args:
            target: Target to be used (must be implemented, otherwise a KeyError is raised)
            subjects: List of subject IDs

        Returns:
            Targets, such as age (float) or sex (binary)

        Examples:
            >>> my_data = ExampleData()
            >>> my_data.load_targets("NotATarget", ["PDUFHPNPU", "P8NE8ER9", "OEEKPOPRRK"])
            Traceback (most recent call last):
            ...
            KeyError: 'The target NotATarget was not recognised'
        """
        if target == "age":
            return self._target_age(subjects=subjects)
        elif target == "sex":
            return self._target_sex(subjects=subjects)
        else:
            raise KeyError(f"The target {target} was not recognised")

    # --------------------------------
    # Implementing a variety of possible
    # targets
    # --------------------------------
    @classmethod
    def _target_sex(cls, subjects: List[str]) -> numpy.array:
        """
        Get the sex of the subjects. Returns 0 if male, 1 if female.
        Args:
            subjects: List of subject IDs

        Returns:
            Sex of the participants as a numpy array. The i-th element in the array corresponds to the i-subject, and
            will be 0 if male, 1 if female

        Examples:
            >>> ExampleData()._target_sex(["KHGCUYGBIUCUYB", "P8NE8ER9","PDUFHPNPU", "OEEKPOPRRK", "BHERIUHIUFRIJ",
            ...                            "OIIUFIGOGD",  "OIHIHUYGO", "ICUYNUIYNPN"])
            array([1, 0, 0, 1, 1, 0, 1, 0])
        """
        # Get dict with subject ID as key and sex as target
        sex_dict = cls.get_subject_sex()

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sex_dict[subject] for subject in subjects])

    @classmethod
    def _target_age(cls, subjects: List[str]) -> numpy.array:
        """
        Get age sex of the subjects
        Args:
            subjects: List of subject IDs

        Returns:
            Age of the participants as a numpy array. The i-th element in the array corresponds to the i-subject

        Examples:
            >>> ExampleData._target_age(["KHGCUYGBIUCUYB", "P8NE8ER9","PDUFHPNPU", "OEEKPOPRRK", "BHERIUHIUFRIJ",
            ...                          "OIIUFIGOGD",  "OIHIHUYGO", "ICUYNUIYNPN"])  # doctest: +NORMALIZE_WHITESPACE
            array([21.82288752, 64.90071275, 51.36861962, 29.01780755, 47.93117405, 80.79343807, 14.8003367 ,
                   81.08120565])
        """
        # Convert to Dict with subject ID as key and sex as target
        age_dict = cls.get_subject_age()

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([age_dict[subject] for subject in subjects])
