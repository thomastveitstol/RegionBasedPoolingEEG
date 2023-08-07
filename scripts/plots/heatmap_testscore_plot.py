"""
Script for generating/plotting heatmaps
"""
from typing import Any, Dict, List, Tuple
import os
import configparser
from configparser import ConfigParser
import re

from matplotlib import pyplot
import numpy
import pandas
import seaborn

from src.config_functions import str_to_list
from src.data.paths import get_continuous_models, get_rbp_mean_models, get_rbp_shared_rocket_models, \
    get_rbp_connected_search_models
from src.utils import get_metrics_single_run, append_values


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


def _get_performance(model_path: str, performance_metric: str, median: bool) -> Dict[str, float]:
    """
    Get the test performance of a model
    Args:
        model_path: Path to model
        performance_metric: Performance metric, e.g. 'auc'

    Returns: Dict with channel system as keys and performance (either median or mean of k-fold runs) as values

    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = get_rbp_shared_rocket_models()[0]
        >>> _get_performance(model_path=my_path, performance_metric="auc", median=False)
        {'Test (c=129)': 0.9468117577749544, 'Test (c=32)': 0.9334182316455962, 'Test (c=65)': 0.9365750674414635}
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

    # Compute point estimate and return
    func = numpy.median if median else numpy.mean
    return {ch_system: func(performance) for ch_system, performance in peak_performances.items()}


# -------------------
# Getter functions for the hyperparameters
# -------------------
def _get_num_channel_splits(config: ConfigParser) -> int:
    """Get the number of channel splits used for a training script"""
    # Return the number of channel splits. If not specified, then Region Based Pooling was not used, and it is therefore
    # set to -1
    if config.has_option("CHANNELSPLITHYPERPARAMS", "num_channel_splits"):
        return config.getint("CHANNELSPLITHYPERPARAMS", "num_channel_splits")

    try:
        topological_regions = str_to_list(config.get("REGIONS2BINS", "topological_regions"), arg_type="int")
    except configparser.NoSectionError:
        return -1
    return len(topological_regions)


def _get_min_nodes(config: ConfigParser) -> int:
    return config.getint("CHANNELSPLITHYPERPARAMS", "min_nodes")


def _get_pooling_module_name(config: ConfigParser) -> int:
    name = config.get("POOLINGMODULEHYPERPARAMS", "name", fallback="str: SharedRocketKernels")[5:]
    str_to_int = {"SharedRocketKernels": 0, "ConnectedSearch2": 1}  # I do not like this solution
    return str_to_int[name]


def _get_hyperparameters(path: str, hyperparameters: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Get the hyeprparameters
    Args:
        path: Path to results file
        hyperparameters: Hyperparameters to get

    Returns: Dict

    Examples:
        >>> from src.data.paths import get_rbp_shared_rocket_models
        >>> my_path = get_rbp_shared_rocket_models()[0]
        >>> my_hyperparameters = ("min_nodes", "num_channel_splits")
        >>> _get_hyperparameters(path=my_path, hyperparameters=my_hyperparameters)
        {'min_nodes': 1, 'num_channel_splits': 25}
    """
    # Read from the config file
    file_names = os.listdir(path)
    config_file_name = [file_name for file_name in file_names if file_name[:5] == "conf_"][0]

    config = ConfigParser()
    config.read(os.path.join(path, config_file_name))

    # -------------------
    # Define dict containing the get-functions
    # -------------------
    mapping = {"min_nodes": _get_min_nodes, "num_channel_splits": _get_num_channel_splits,
               "pooling_module_name": _get_pooling_module_name}

    # -------------------
    # Loop though all hyperparameters
    # -------------------
    model_settings: Dict[str, Any] = dict()
    for hyperparam in hyperparameters:
        model_settings[hyperparam] = mapping[hyperparam](config=config)

    return model_settings


def main() -> None:
    # Set the path where the plots are going to be stored. It must contain four folders, named 'continuous', 'mean',
    # 'shared_rocket', and 'connected'
    save_path = "/home/thomas/Documents/Paper1/grid_search"

    # Which model to plot. Set to 'continuous', 'mean', 'shared_rocket', or 'connected', depending on which results you
    # want to plot
    model_to_plot = "shared_rocket"

    # Define variables
    variables = "min_nodes", "num_channel_splits"

    # ----------------------
    # Runs when in Trondheim
    # ----------------------
    models = {"continuous": get_continuous_models(),
              "mean": get_rbp_mean_models(),
              "shared_rocket": get_rbp_shared_rocket_models(),
              "connected": get_rbp_connected_search_models()}[model_to_plot]

    # Select performance metrics
    performance_metric = "auc"

    # Use median or mean of the k-fold results
    median = False

    # Title using the terms of the paper
    title = {"mean": "Averaging",
             "shared_rocket": "Channel Attention",
             "connected": "Channel Attention with Head Region",
             "continuous": "Continuous Attention"}[model_to_plot]

    # Loop through all experiments
    data_samples: Dict[str, Dict[str, Any]] = dict()  # {Channel system: {Variable name: variable value}}
    for model_path in models:
        # ----------------------
        # Get hyperparameters and performance
        # ----------------------
        model_hyperparameters = _get_hyperparameters(path=model_path, hyperparameters=variables)
        model_performance = _get_performance(model_path=model_path, performance_metric=performance_metric,
                                             median=median)

        # ----------------------
        # Add instance
        # ----------------------
        for channel_system, performance in model_performance.items():
            data_instance = model_hyperparameters  # don't really care if mutable
            data_instance["Performance"] = model_performance[channel_system]
            main_dict = data_samples[channel_system] if channel_system in data_samples else None
            data_samples[channel_system] = append_values(main_dict=main_dict, append_dict=data_instance)

    # ----------------------
    # Plotting
    # ----------------------
    # Convert to pandas dataframe
    data = {ch_system: pandas.DataFrame(data_row) for ch_system, data_row in data_samples.items()}

    font_size = 50

    # Plot all channel systems
    for ch_system, df in data.items():
        # Pivot the dataframe to create a matrix of performance values for each combination of hyperparameters
        heatmap_data = df.pivot(index="min_nodes", columns="num_channel_splits", values="Performance")

        # Create the heatmap with numerical values
        fig, ax = pyplot.subplots(figsize=(26, 13))
        seaborn.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="coolwarm", vmin=0.86, vmax=0.98,
                        cbar_kws={"label": "Model Performance"}, ax=ax, annot_kws={"size": font_size})

        # Set the font size of the color bar labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.set_ylabel(performance_metric.upper(), fontsize=font_size)

        # Set axis labels and title
        ax.set_xlabel("Number of channel splits", fontsize=font_size)
        ax.set_ylabel("Minimum number of electrodes", fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        ax.tick_params("y", labelsize=font_size)
        fig.suptitle(title, fontsize=font_size+5, fontweight="bold")
        ax.set_title(f"Model Performance, {ch_system}", fontsize=font_size)

        num_channels = re.split("[()]", ch_system)[1][2:]
        fig.savefig(os.path.join(save_path, model_to_plot, f"{model_to_plot}_{num_channels}"))


if __name__ == "__main__":
    main()
