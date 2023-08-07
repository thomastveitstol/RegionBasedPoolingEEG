import numpy

from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem, Reduced1CleanedChildChannelSystem, \
    Reduced3CleanedChildChannelSystem
from src.data.datasets.data_base import smallest_channel_system, get_illegal_channels
from src.data.datasets.example_data import ExampleData


def test_k_fold_sex_split() -> None:
    num_folds = 3

    # --------------------------------
    # Load dataset object
    # --------------------------------
    dataset = ExampleData()

    # Get all the subject IDs
    subject_ids = tuple(dataset.get_subject_ids())

    # Get the dictionary of the train/val split
    numpy.random.seed(3)
    data_split = dataset.k_fold_sex_split(subjects=subject_ids, num_folds=num_folds, force_balanced=True,
                                          num_subjects=None)

    # --------------------------------
    # Test
    # --------------------------------
    assert len(data_split) == num_folds, "Number of folds was incorrect"

    assert data_split == (('ICUYNUIYNPN', 'UYNVCNNNNHFHK', 'OIHIHUYGO', 'KHGCUYGBIUCUYB'),
                          ('UIYNCOIUERIOPCU', 'BHERIUHIUFRIJ'), ('P8NE8ER9', 'OEEKPOPRRK'))


# -----------------------
# Test functions
# -----------------------
def test_smallest_channel_system() -> None:
    """Test if the smallest channel system is returned by the smallest_channel_system() function"""
    # -------------------
    # Define some channel systems
    # -------------------
    channel_systems = [CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem()]

    # -------------------
    # Verify that the smallest channel
    # system is returned
    # -------------------
    expected = Reduced3CleanedChildChannelSystem()
    actual = smallest_channel_system(channel_systems=channel_systems)

    # Just name checking, as __eq__ is not overridden
    assert expected.name == actual.name, f"The expected channel system {expected.name} did not match the actual " \
                                         f"channel system {actual.name}"


def test_illegal_indices() -> None:
    """Test if the illegal indices are correctly returned by the get_illegal_indices function"""
    # -------------------
    # Define main channel system and
    # some reduced channel systems
    # -------------------
    original_system = CleanedChildChannelSystem()
    reduced_systems = (Reduced3CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       CleanedChildChannelSystem())

    # -------------------
    # Get the illegal channel indices
    # -------------------
    illegal_indices = get_illegal_channels(main_channel_system=original_system, reduced_channel_systems=reduced_systems)

    # Test if the keys are as expected (the channel system names):
    actual_channel_names = list(illegal_indices.keys())
    expected_channel_names = [Reduced3CleanedChildChannelSystem().name,
                              Reduced1CleanedChildChannelSystem().name,
                              CleanedChildChannelSystem().name]
    assert actual_channel_names == expected_channel_names, f"Expected the following channel system names: " \
                                                           f"{expected_channel_names}, but received " \
                                                           f"{actual_channel_names}"

    # Test if the values are as expected:
    actual_channels_1 = illegal_indices["Reduced1CleanedChildMindInstitute"]
    expected_channels_1 = ['E2', 'E5', 'E7', 'E8', 'E10', 'E12', 'E14', 'E15', 'E17', 'E18', 'E20', 'E21', 'E25',
                           'E26', 'E31', 'E35', 'E38', 'E39', 'E40', 'E42', 'E43', 'E48', 'E49', 'E50', 'E53', 'E54',
                           'E55', 'E56', 'E59', 'E61', 'E63', 'E65', 'E66', 'E68', 'E69', 'E71', 'E73', 'E74', 'E76',
                           'E78', 'E79', 'E80', 'E81', 'E82', 'E84', 'E86', 'E88', 'E89', 'E90', 'E91', 'E93', 'E94',
                           'E99', 'E101', 'E106', 'E107', 'E109', 'E110', 'E113', 'E115', 'E118', 'E119', 'E120',
                           'E121']

    actual_channels_2 = illegal_indices["Reduced3CleanedChildMindInstitute"]
    expected_channels_2 = ['E1', 'E2', 'E3', 'E4', 'E5', 'E7', 'E8', 'E10', 'E12', 'E13', 'E14', 'E16', 'E17', 'E18',
                           'E19', 'E20', 'E21', 'E23', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E34',
                           'E35', 'E37', 'E38', 'E39', 'E40', 'E41', 'E42', 'E44', 'E46', 'E47', 'E48', 'E49', 'E50',
                           'E51', 'E53', 'E54', 'E55', 'E57', 'E59', 'E60', 'E61', 'E63', 'E64', 'E65', 'E66', 'E67',
                           'E69', 'E71', 'E72', 'E73', 'E74', 'E76', 'E77', 'E78', 'E79', 'E80', 'E81', 'E82', 'E84',
                           'E85', 'E86', 'E87', 'E88', 'E89', 'E90', 'E91', 'E93', 'E95', 'E97', 'E98', 'E99', 'E100',
                           'E101', 'E102', 'E103', 'E105', 'E106', 'E109', 'E110', 'E111', 'E112', 'E113', 'E114',
                           'E115', 'E116', 'E117', 'E118', 'E119', 'E121', 'E123']

    actual_channels_3 = illegal_indices["CleanedChildMindInstitute"]
    expected_channels_3 = []

    assert actual_channels_1 == expected_channels_1, f"Expected the following illegal channel indices " \
                                                     f"from Reduced1CleanedChildChannelSystem: " \
                                                     f"{expected_channels_1}, but received " \
                                                     f"{actual_channels_1}"
    assert actual_channels_2 == expected_channels_2, f"Expected the following illegal channel indices " \
                                                     f"from Reduced3CleanedChildChannelSystem: " \
                                                     f"{expected_channels_2}, but received " \
                                                     f"{actual_channels_2}"
    assert actual_channels_3 == expected_channels_3, f"Expected the following illegal channel indices " \
                                                     f"from CleanedChildChannelSystem: " \
                                                     f"{expected_channels_3}, but received " \
                                                     f"{actual_channels_3}"
