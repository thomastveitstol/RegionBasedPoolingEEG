"""
Script for selecting the best RBP, interpolation, and zero-filling models. It is selected by maximising the performance
on the validation set (highest sum of performances on the three channel systems, measured by AUC).
"""
import os
from typing import Dict, List, Tuple

import numpy

from src.data.paths import get_rbp_connected_search_models, get_rbp_shared_rocket_models, get_rbp_mean_models, \
    get_continuous_models, get_interpolation_models, get_zero_fill_models
from src.utils import get_metrics_single_run


def _ch_system_to_label(ch_system: str) -> str:
    """Get a better looking label, given the name of a channel system"""
    # Define some good-looking labels
    mapping = {"test_cleanedchildmindinstitute": "Test (c=129)",
               "test_reduced1cleanedchildmindinstitute": "Test (c=65)",
               "test_reduced3cleanedchildmindinstitute": "Test (c=32)"}

    # Use them, if possible
    try:
        return mapping[ch_system]
    except KeyError:
        return ch_system


def _get_test_performance(model_path: str, performance_metric: str) -> Dict[str, List[float]]:
    """
    Get the test performance of a model
    Args:
        model_path: Path to model
        performance_metric: Performance metric, e.g. 'auc'

    Returns: Dict with channel system as keys and performance (either median or mean of k-fold runs) as values

    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = get_rbp_shared_rocket_models()[0]
        >>> _get_test_performance(model_path=my_path, performance_metric="auc")  # doctest: +NORMALIZE_WHITESPACE
        {'Test (c=129)': [0.9580537436409601, 0.9549639524359415, 0.9404825067881776, 0.9327105163489458,
                          0.9478480696607471],
         'Test (c=32)': [0.9446958584313848, 0.9286539121750257, 0.9189475983895633, 0.9307221310440601,
                         0.9440716581879467],
         'Test (c=65)': [0.9555257326550358, 0.9421678474454604, 0.9331793639399519, 0.9187918192147455,
                         0.9332105739521238]}
    """
    # Get folders name of all runs
    sub_folders = os.listdir(model_path)
    sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder[:4] == "Run_"]

    # Loop through all runs
    peak_performances: Dict[str, List[float]] = dict()
    for run in sub_folders:
        # Get all metrics from the run
        histories = get_metrics_single_run(path=os.path.join(model_path, run))

        # Get performance on all channel systems
        for channel_system, history in histories.items():
            if channel_system[:5] != "test_":
                continue

            # Compute performance
            test_performance = history[performance_metric][0]

            # Store in dict
            label = _ch_system_to_label(channel_system)
            if label in peak_performances:
                peak_performances[label].append(test_performance)
            else:
                peak_performances[label] = [test_performance]

    # Return performance on all folds
    return peak_performances


def _get_val_performance(model_path: str, performance_metric: str, median: bool = False) -> float:
    """
    Get the validation performance of a model
    Args:
        model_path: Path to model
        performance_metric: Performance metric, e.g. 'auc'
        median: To use median (True) or mean (False)

    Returns: The average validation performance on the channel systems which epoch maximise the sum of performances

    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = get_rbp_shared_rocket_models()[0]
        >>> _get_val_performance(model_path=my_path, performance_metric="auc")
        0.9398985588314895
    """
    # Get folders name of all runs
    sub_folders = os.listdir(model_path)
    sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder[:4] == "Run_"]

    # Loop through all runs
    peak_performances: List[float] = list()
    for run in sub_folders:
        # Get all metrics from the run
        histories = get_metrics_single_run(path=os.path.join(model_path, run))

        # Get performance on all channel systems
        performances = list()
        for channel_system, history in histories.items():
            if channel_system[:4] != "val_":
                continue

            # Get the entire validation curve
            performances.append(numpy.expand_dims(history[performance_metric], axis=1))

        # Compute performance per epoch as the mean on all channel systems
        performances = numpy.concatenate(performances, axis=1)
        assert performances.shape[-1] == 3, f"Expected three channel systems, but found {performances.shape[-1]}"
        performances = numpy.mean(performances, axis=1)

        # Compute the max performance
        peak_performance = max(performances)

        # Store
        peak_performances.append(peak_performance)

    # Compute point estimate and return
    func = numpy.median if median else numpy.mean
    return float(func(peak_performances))


def _select_best_model(model_paths: Tuple[str, ...], performance_metric: str, median: bool = False) -> str:
    """
    Get the highest performing model, based on the validation
    Args:
        model_paths: Model folder names
        performance_metric: Performance metric
        median: To use median (True) or mean (False)

    Returns: The index of the best performing model, and the performance

    Examples:
        >>> my_models = get_rbp_connected_search_models() + get_rbp_mean_models() + get_rbp_shared_rocket_models()
        >>> _select_best_model(model_paths=my_models, performance_metric="auc")  # doctest: +ELLIPSIS
        '.../5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_11:14:26'
    """
    performances = tuple(_get_val_performance(model_path=path, performance_metric=performance_metric,
                                              median=median) for path in model_paths)
    return model_paths[int(numpy.argmax(performances))]


def main() -> None:
    # Define all models
    rbp_models = get_rbp_connected_search_models() + get_rbp_shared_rocket_models() + get_rbp_mean_models() + \
                 get_continuous_models()
    interpolation_models = get_interpolation_models()
    zero_fill_models = get_zero_fill_models()

    # Select performance metrics
    performance_metric = "auc"

    # To print the best model
    print_best_models = True

    # Use median or mean of the k-fold results
    median = False

    # Select the best model
    best_rbp_model = _select_best_model(model_paths=rbp_models, performance_metric=performance_metric, median=median)
    best_ssi_model = _select_best_model(model_paths=interpolation_models, performance_metric=performance_metric,
                                        median=median)
    best_zf_model = _select_best_model(model_paths=zero_fill_models, performance_metric=performance_metric,
                                       median=median)

    # Maybe print best models
    if print_best_models:
        print(f"Best RBP: {best_rbp_model}")
        print(f"Best SSI: {best_ssi_model}")
        print(f"Best ZF: {best_zf_model}")

    # Compute the test performances of the selected models
    rbp_performances = _get_test_performance(model_path=best_rbp_model, performance_metric=performance_metric)
    ssi_performances = _get_test_performance(model_path=best_ssi_model, performance_metric=performance_metric)
    zf_performances = _get_test_performance(model_path=best_zf_model, performance_metric=performance_metric)

    # Print the averages
    for performances, method in zip((rbp_performances, ssi_performances, zf_performances), ("RBP", "SSI", "ZF")):
        for channel_system, metrics in performances.items():
            print(f"{method}, {channel_system}: {100 * numpy.mean(metrics):.2f}%")


if __name__ == "__main__":
    main()
