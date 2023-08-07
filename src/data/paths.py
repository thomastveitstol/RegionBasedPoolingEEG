"""
File for getting the path to the results of the runs
"""
import os
from typing import Tuple


def get_results_dir() -> str:
    """
    Get the path of the results folder

    Returns:
        The path to the results folder

    Examples:
        >>> get_results_dir()  # doctest: +ELLIPSIS
        '.../RegionBasedPoolingEEG/src/data/results'
    """
    return os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------
# Baseline models
# ---------------------------------
def get_interpolation_models() -> Tuple[str, ...]:
    # Models
    models = ("5_Fold_FixedChannels_Interpolation_Inception_2023-06-30_02:14:46",
              "5_Fold_FixedChannels_Interpolation_Inception_2023-06-30_03:27:32",
              "5_Fold_FixedChannels_Interpolation_Inception_2023-06-30_04:38:10",
              "5_Fold_FixedChannels_Interpolation_Inception_2023-06-30_05:48:13")

    # Add results path and return
    return attach_results_path(models)


def get_zero_fill_models() -> Tuple[str, ...]:
    # Models
    models = ("5_Fold_FixedChannels_ZeroFill_Inception_2023-06-30_06:56:52",
              "5_Fold_FixedChannels_ZeroFill_Inception_2023-06-30_08:03:11",
              "5_Fold_FixedChannels_ZeroFill_Inception_2023-06-30_09:07:02",
              "5_Fold_FixedChannels_ZeroFill_Inception_2023-06-30_10:10:22")

    # Add results path and return
    return attach_results_path(models)


# ---------------------------------
# Region based pooling models
# ---------------------------------
def get_rbp_connected_search_models() -> Tuple[str, ...]:
    # Models
    models = ("5Fold_RBP_Shared_ConnectedSearch2_2023-06-27_17:34:00",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-27_21:36:35",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-28_00:10:14",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-28_17:11:32",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_05:56:43",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_10:01:26",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_16:44:50",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_18:59:29",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_20:37:36",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_22:03:59",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-29_23:42:42",
              "5Fold_RBP_Shared_ConnectedSearch2_2023-06-30_01:03:02")

    # Add results path and return
    return attach_results_path(models)


def get_rbp_shared_rocket_models() -> Tuple[str, ...]:
    # Models
    models = ("5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_11:14:26",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_13:44:08",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_15:32:47",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_17:04:56",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_21:10:02",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_23:41:14",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_01:40:01",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_03:16:48",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_04:38:26",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_05:55:23",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_07:14:58",
              "5Fold_RBP_Shared_SharedRocketKernels_2023-07-01_08:28:39")

    # Add results path and return
    return attach_results_path(models)


def get_rbp_mean_models() -> Tuple[str, ...]:
    # Models
    models = ("5Fold_RBP_Mean_2023-07-01_09:39:48",
              "5Fold_RBP_Mean_2023-07-01_10:55:19",
              "5Fold_RBP_Mean_2023-07-01_12:00:07",
              "5Fold_RBP_Mean_2023-07-01_13:02:17",
              "5Fold_RBP_Mean_2023-07-01_14:38:03",
              "5Fold_RBP_Mean_2023-07-01_15:54:32",
              "5Fold_RBP_Mean_2023-07-01_17:06:43",
              "5Fold_RBP_Mean_2023-07-01_18:07:42",
              "5Fold_RBP_Mean_2023-07-01_19:04:41",
              "5Fold_RBP_Mean_2023-07-01_20:00:41",
              "5Fold_RBP_Mean_2023-07-01_20:56:56",
              "5Fold_RBP_Mean_2023-07-01_21:51:35")

    # Add results path and return
    return attach_results_path(models)


def get_continuous_models() -> Tuple[str, ...]:
    # Models
    models = ("5Fold_RBP_ContinuousAttention_2023-07-01_22:45:56",
              "5Fold_RBP_ContinuousAttention_2023-07-02_20:30:44",
              "5Fold_RBP_ContinuousAttention_2023-07-03_17:15:28",
              "5Fold_RBP_ContinuousAttention_2023-07-04_16:18:26",
              "5Fold_RBP_ContinuousAttention_2023-07-06_12:35:18",
              "5Fold_RBP_ContinuousAttention_2023-07-08_08:02:07",
              "5Fold_RBP_ContinuousAttention_2023-07-10_01:30:00",
              "5Fold_RBP_ContinuousAttention_2023-07-15_07:38:04",
              "5Fold_RBP_ContinuousAttention_2023-07-20_07:48:11")

    # Add results path and return
    return attach_results_path(models)


# -----------------------
# Functions
# -----------------------
def attach_results_path(model_names: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Attaches the path of the results dir to all model names
    Args:
        model_names: Model names

    Returns: Path to model results files

    Examples:
        >>> attach_results_path(model_names=("m1", "m2", "m3"))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ('.../RegionBasedPoolingEEG/src/data/results/m1',
         '.../RegionBasedPoolingEEG/src/data/results/m2',
         '.../RegionBasedPoolingEEG/src/data/results/m3')

    """
    results_path = get_results_dir()
    return tuple(os.path.join(results_path, model) for model in model_names)
