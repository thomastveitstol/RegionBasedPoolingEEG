"""
Classes for Channel Systems and Dataset for the self-cleaned version of the ChildMindInstitute dataset (pipeline from
Christoffer Hatlestad-Hall).

Please see the download script for child mind data in scripts folder, to download the dataset. The cleaning pipeline is
available at https://github.com/hatlestad-hall/prep-childmind-eeg
"""
import os
from typing import Dict, List, Optional

import mne
import enlighten
import numpy
import pandas
import pickle

from src.utils import CartesianCoordinates
from src.data.datasets.data_base import BaseChannelSystem, ReducedBaseChannelSystem, EEGDataset


# ------------------------------------
# Channel systems
# ------------------------------------
# Main channel system
class CleanedChildChannelSystem(BaseChannelSystem):

    def __init__(self):
        """
        Examples:
            >>> CleanedChildChannelSystem()  # doctest: +NORMALIZE_WHITESPACE
            --- Channel System ---
            Name: CleanedChildMindInstitute
            Number of channels: 129
        """
        super().__init__(name="CleanedChildMindInstitute", num_channels=129)

    @staticmethod
    def get_electrode_positions() -> Dict[str, CartesianCoordinates]:
        """
        Examples:
            >>> CleanedChildChannelSystem.get_electrode_positions()["Cz"]
            CartesianCoordinates(coordinates=(6.123233995736766e-17, -0.0, 1.0))
            >>> CleanedChildChannelSystem.get_electrode_positions()["E3"]
            CartesianCoordinates(coordinates=(0.839869834644161, -0.42510579491903916, 0.33749625772894387))
            >>> len(CleanedChildChannelSystem.get_electrode_positions())
            129
        """
        # Load previously saved channel positions
        # todo: this path is hard-coded, and must be changed
        with open("/home/thomas/ChildMindInstituteCleaned/channel_positions.pkl", "rb") as f:
            channel_positions = pickle.load(f)["ch_pos"]

        return {channel_name: CartesianCoordinates(tuple(numpy.array((y, -x, z))/numpy.linalg.norm((x, y, z))))
                for channel_name, (x, y, z) in channel_positions.items()}

    def channel_name_to_index(self) -> Dict[str, int]:
        """
        Examples:
            >>> CleanedChildChannelSystem().channel_name_to_index()["Cz"]
            128
            >>> list(CleanedChildChannelSystem().channel_name_to_index().keys())[:15]
            ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15']
        """
        channel_names = self.get_electrode_positions()
        return {channel_name: i for i, channel_name in enumerate(channel_names)}


# Reduced channel systems
class Reduced1CleanedChildChannelSystem(ReducedBaseChannelSystem):
    """
    Selects the 64 (+ Cz) electrodes which are present in the 64-channel version of HydroCel GSN

    See page 124 and 125 in https://www.documents.philips.com/assets/20180705/6f388e7ade4d41e38ad5a91401755b6f.pdf for
    channel names
    """

    original_channel_system = CleanedChildChannelSystem()
    channel_names = ("E1", "E3", "E4", "E6", "E9", "E11", "E13", "E16", "E19", "E22", "E23", "E24", "E27", "E28", "E29",
                     "E30", "E32", "E33", "E34", "E36", "E37", "E41", "E44", "E45", "E46", "E47", "E51", "E52", "E57",
                     "E58", "E60", "E64", "E67", "E62", "E70", "E72", "E75", "E77", "E83", "E85", "E87", "E92", "E95",
                     "E96", "E97", "E98", "E100", "E102", "E103", "E104", "E105", "E108", "E111", "E112", "E114",
                     "E116", "E117", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "Cz")

    def __init__(self):
        """
        Examples:
            >>> Reduced1CleanedChildChannelSystem()
            --- Channel System ---
            Name: Reduced1CleanedChildMindInstitute
            Number of channels: 65
            >>> Reduced1CleanedChildChannelSystem().original_channel_system
            --- Channel System ---
            Name: CleanedChildMindInstitute
            Number of channels: 129
        """
        super().__init__(reduced_name="Reduced1", num_channels=len(self.channel_names))

    @classmethod
    def get_electrode_positions(cls) -> Dict[str, CartesianCoordinates]:
        """
        todo: consider making this default in base class
        Examples:
            >>> my_channel_system = Reduced1CleanedChildChannelSystem()
            >>> list(my_channel_system.get_electrode_positions().keys())[:10]
            ['E1', 'E3', 'E4', 'E6', 'E9', 'E11', 'E13', 'E16', 'E19', 'E22']
            >>> Reduced1CleanedChildChannelSystem.get_electrode_positions()["E1"]
            CartesianCoordinates(coordinates=(0.6569640894891879, -0.6887141049559512, -0.30671006953858787))
            >>> len(Reduced1CleanedChildChannelSystem.get_electrode_positions())
            65
        """
        # Get all electrode positions (of the original channel system)
        all_electrode_positions = cls.original_channel_system.get_electrode_positions()

        # Return only the correct ones
        return {ch_name: position for i, (ch_name, position) in enumerate(all_electrode_positions.items())
                if ch_name in cls.channel_names}


class Reduced3CleanedChildChannelSystem(ReducedBaseChannelSystem):
    """
    Selects the 32 electrodes which are present in the 32-channel version of HydroCel GSN

    See page 123 and 125 in https://www.documents.philips.com/assets/20180705/6f388e7ade4d41e38ad5a91401755b6f.pdf for
    channel names

    (Pardon the name of the class, I formerly had a class called 'Reduced2CleanedChildChannelSystem', which selected a
    third of the channels. However, during the development and implementation, I (together with my supervisors) figured
    it was better to use the 32 and 65 channel system equivalents only, as they actually exist. It was simply easier to
    keep the current class name than to change everywhere in the code)
    """

    original_channel_system = CleanedChildChannelSystem()
    channel_names = ("E22", "E9", "E24", "E124", "E36", "E104", "E52", "E92", "E70", "E83", "E33", "E122", "E45",
                     "E108", "E58", "E96", "E11", "Cz", "E62", "E75", "E43", "E120", "E56", "E107", "E68", "E94",
                     "E15", "E6", "E127", "E126", "E128", "E125")

    def __init__(self):
        """
        Examples:
            >>> Reduced3CleanedChildChannelSystem()
            --- Channel System ---
            Name: Reduced3CleanedChildMindInstitute
            Number of channels: 32
            >>> Reduced3CleanedChildChannelSystem().original_channel_system
            --- Channel System ---
            Name: CleanedChildMindInstitute
            Number of channels: 129
        """
        super().__init__(reduced_name="Reduced3", num_channels=len(self.channel_names))

    @classmethod
    def get_electrode_positions(cls) -> Dict[str, CartesianCoordinates]:
        """
        Examples:
            >>> my_channel_system = Reduced3CleanedChildChannelSystem()
            >>> list(my_channel_system.get_electrode_positions().keys())[:10]
            ['E6', 'E9', 'E11', 'E15', 'E22', 'E24', 'E33', 'E36', 'E43', 'E45']
            >>> Reduced3CleanedChildChannelSystem.get_electrode_positions()["E22"]
            CartesianCoordinates(coordinates=(0.9504267034248264, 0.2883328336962804, 0.11641846258085796))
            >>> len(Reduced3CleanedChildChannelSystem.get_electrode_positions())
            32
        """
        # Get all electrode positions (of the original channel system)
        all_electrode_positions = cls.original_channel_system.get_electrode_positions()

        # Keep every second channel
        return {ch_name: position for i, (ch_name, position) in enumerate(all_electrode_positions.items())
                if ch_name in cls.channel_names}


# ------------------------------------
# Dataset
# ------------------------------------
class CleanedChildData(EEGDataset):
    """
    The dataset from Child Mind Institute, clean using a pipeline by Christoffer Hatlestad-Hall
    """

    def __init__(self, root_dir="/home/thomas/ChildMindInstituteCleaned"):
        """
        Initialise
        Args:
            root_dir: Root directory
        Examples:
            >>> CleanedChildData()
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
        """
        super().__init__(root_dir=root_dir, max_time_steps=15000, channel_system=CleanedChildChannelSystem())

    # --------------------------------
    # Methods for subject related info
    # --------------------------------
    def get_subject_ids(self) -> List[str]:
        """
        Get subject IDs of all available subjects
        Returns:
            List of all subject IDs
        Examples:
            >>> my_dataset = CleanedChildData()
            >>> my_dataset.get_subject_ids()[:5], len(my_dataset.get_subject_ids())
            (['NDARJK651XB0', 'NDARFK452XCU', 'NDARUR298LVX', 'NDARMV718DYL', 'NDARFJ179MG0'], 2749)
        """
        # Get the IDs in the path folder:
        subject_ids = os.listdir(f"{self.root_dir}/numpy_arrays")

        # Remove .npy:
        subject_ids = [subject_id[:-4] for subject_id in subject_ids]

        return subject_ids

    def _get_participant_info(self) -> numpy.ndarray:
        """
        Get info of the participants. Note that the paths are hard coded.
        Returns:
            Info of the participants. See the raw files for details

        Examples:
            >>> CleanedChildData()._get_participant_info()[:5]  # doctest: +NORMALIZE_WHITESPACE
            array([[1, 'NDARAA075AMK', 'R1', 1.0, 6.72804, 65.57, 'No', 'Yes'],
                   [2, 'NDARAA112DMH', 'R1', 0.0, 5.545744, 40.02, 'No', 'Yes'],
                   [3, 'NDARAA117NEJ', 'R1', 0.0, 7.475929, 62.23, 'No', 'Yes'],
                   [4, 'NDARAA306NT2', 'R8', 1.0, 21.216746, 6.67, 'Yes', 'No'],
                   [5, 'NDARAA396TWZ', 'R10', 0.0, 7.05989, 55.63, 'Yes', 'No']], dtype=object)
        """
        # Load the .tsv file containing information regarding the participants
        participant_info = pandas.read_csv(f"{self.root_dir}/participants.tsv", sep="\t").to_numpy()
        return participant_info

    def get_subject_sex(self) -> Dict[str, int]:
        """
        Get a Dict containing the sex of all subjects available
        Returns: Dict with subject ID (str) as keys and
        Examples:
            >>> list(CleanedChildData().get_subject_sex().values())[:5]
            [1, 0, 0, 1, 0]
            >>> list(CleanedChildData().get_subject_sex().keys())[:5]
            ['NDARAA075AMK', 'NDARAA112DMH', 'NDARAA117NEJ', 'NDARAA306NT2', 'NDARAA396TWZ']
        """
        # Load the .csv files containing information regarding the participants as
        data = self._get_participant_info()
        data = data[:, [1, 3]]  # Extract IDs and sex only

        # Convert to Dict with subject ID as key and sex as target
        sex_dict = {sub_id: int(sex) for sub_id, sex in data if not numpy.isnan(sex)}
        return sex_dict

    def get_subject_age(self) -> Dict[str, float]:
        """
        Get a Dict containing the sex of all subjects available
        Returns: Dict with subject ID (str) as keys and
        Examples:
            >>> list(CleanedChildData().get_subject_age().values())[:5]
            [6.72804, 5.545744, 7.475929, 21.216746, 7.05989]
            >>> list(CleanedChildData().get_subject_age().keys())[:5]
            ['NDARAA075AMK', 'NDARAA112DMH', 'NDARAA117NEJ', 'NDARAA306NT2', 'NDARAA396TWZ']
        """
        # Load the .csv files containing information regarding the participants as
        data = self._get_participant_info()
        data = data[:, [1, 4]]  # Extract IDs and age only

        # Convert to Dict with subject ID as key and age as target
        age_dict = {sub_id: age for sub_id, age in data}
        return age_dict

    # --------------------------------
    # Methods for loading data
    # --------------------------------
    # EEG Data
    def load_eeg_data(self, subjects: List[str],  time_series_start: int = 0,
                      truncate: int = 5000) -> numpy.ndarray:
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
            >>> my_data = CleanedChildData()
            >>> my_data.load_eeg_data(subjects=['NDARJK651XB0', 'NDARFK452XCU', 'NDARUR298LVX'], time_series_start=1000,
            ...                       truncate=4000).shape
            (3, 129, 4000)
        """
        # -------------------------
        # Check inputs
        # -------------------------
        self.check_inputs_load_eeg_data(time_series_start=time_series_start, truncate=truncate)

        # -------------------------
        # Loop through and load data
        # -------------------------
        # Initialise numpy ndarray. Will have shape=(participants, channels, time_steps)
        data = numpy.zeros(shape=(len(subjects), self.num_channels, truncate))

        # Set counter
        pbar = enlighten.Counter(total=len(subjects), desc="Loading", unit="subjects")

        for i, sub_id in enumerate(subjects):
            # Get path and load data of the subject
            path = f"{self.root_dir}/numpy_arrays/{sub_id}.npy"
            next_data = numpy.load(path)

            # Truncate, if desired
            if truncate is not None:
                next_data = next_data[..., time_series_start:(truncate + time_series_start)]

            # Add the data
            data[i] = next_data
            pbar.update()

        return data

    # Main method for loading targets
    def load_targets(self, target: str, subjects: List[str]) -> numpy.ndarray:
        """
        Method for loading targets
        Args:
            target: Target to be used (must be implemented, otherwise a KeyError is raised)
            subjects: List of subject IDs

        Returns:
            Targets, such as age (float) or sex (binary)

        Examples:
            >>> my_data = CleanedChildData()
            >>> my_data.load_targets("age", ['NDARJK651XB0', 'NDARFK452XCU', 'NDARFJ179MG0'])
            array([ 6.522701, 16.997718, 14.367214])
            >>> my_data.load_targets("sex", ['NDARJK651XB0', 'NDARFK452XCU', 'NDARFJ179MG0'])
            array([1, 0, 1])
        """
        if target == "age":
            return self._target_age(subjects=subjects)
        elif target == "sex":
            return self._target_sex(subjects=subjects)
        else:
            raise KeyError(f"The target {target} was not recognised")

    # Implementing a variety of possible targets.
    def _target_sex(self, subjects: List[str]) -> numpy.array:
        """
        Get the sex of the subjects. Returns 0 if male, 1 if female.
        Args:
            subjects: List of subject IDs

        Returns:
            Sex of the participants as a numpy array. The i-th element in the array corresponds to the i-subject, and
            will be 0 if male, 1 if female

        Examples:
            >>> CleanedChildData()._target_sex(['NDARJK651XB0', 'NDARFK452XCU', 'NDARFJ179MG0'])
            array([1, 0, 1])
        """
        # Get sex split of all subjects
        sex_dict = self.get_subject_sex()

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([sex_dict[subject] for subject in subjects])

    def _target_age(self, subjects: List[str]) -> numpy.array:
        """
        Get age sex of the subjects
        Args:
            subjects: List of subject IDs

        Returns:
            Age of the participants as a numpy array. The i-th element in the array corresponds to the i-subject

        Examples:
            >>> CleanedChildData()._target_age(['NDARJK651XB0', 'NDARFK452XCU', 'NDARFJ179MG0'])
            array([ 6.522701, 16.997718, 14.367214])
        """
        # Get ages of all subjects
        age_dict = self.get_subject_age()

        # Select the ones passed in the subjects list, and return as numpy array
        return numpy.array([age_dict[subject] for subject in subjects])

    # ---------------
    # For saving
    # ---------------
    @staticmethod
    def save_eeg_data_as_numpy(from_root_path: str, to_root_path: Optional[str] = None,
                               num_time_steps: Optional[int] = None,
                               time_series_start: Optional[int] = None) -> None:
        """
        Method for saving the data to numpy arrays from .set files.
        Args:
            from_root_path: Root path, where the data is stored
            to_root_path: Path of where to store the numpy arrays
            num_time_steps: Number of time steps to save
            time_series_start: Which time step to start from

        Returns: Nothing, it just saves data as numpy arrays

        """
        # Get the IDs in the path folder:
        subjects = os.listdir(from_root_path)

        # Keep only .set files
        subjects = [subject[:-4] for subject in subjects if subject[-4:] == ".set"]

        # Loop through all selected participants
        for subject in subjects:
            # Load data using MNE
            raw_mne = mne.io.read_raw_eeglab(f"{from_root_path}/{subject}.set", verbose=False, preload=True)

            # Convert to numpy
            data = raw_mne.get_data()

            # Maybe truncate
            save = True
            if num_time_steps is not None:
                if num_time_steps-time_series_start <= data.shape[-1]:
                    data = data[:, time_series_start:(time_series_start+num_time_steps)]
                else:
                    save = False
                    print(
                        f"The EEG of subject {subject} did not contain enough time steps and is therefore excluded")

            # If no errors are made, save the data
            if save:
                print(f"Successfully saved EEG data of participant: {subject}")
                numpy.save(f"{to_root_path}/{subject}.npy", arr=data)
            else:
                print(f"Did not saved EEG data of participant: {subject}")
