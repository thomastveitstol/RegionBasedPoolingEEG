"""
Violin plot of test results. It compares Region based pooling, spherical spline interpolation, and zero-filling
"""
import configparser
from configparser import ConfigParser
import os
from typing import Any, Dict, List

from matplotlib import pyplot
from matplotlib import rcParams
import pandas
import seaborn

from src.config_functions import str_to_list
from src.data.paths import get_results_dir
from src.utils import get_metrics_single_run


def _get_num_channel_splits(path: str) -> int:
    """Get the number of channel splits used for a training script"""
    # Read from the config file
    file_names = os.listdir(path)
    config_file_name = [file_name for file_name in file_names if file_name[:5] == "conf_"][0]

    config = ConfigParser()
    config.read(os.path.join(path, config_file_name))

    # Return the number of channel splits epochs. If not specified, then Region Based Pooling was not used, and it is
    # therefore set to -1
    if config.has_option("CHANNELSPLITHYPERPARAMS", "num_channel_splits"):
        return config.getint("CHANNELSPLITHYPERPARAMS", "num_channel_splits")

    try:
        topological_regions = str_to_list(config.get("REGIONS2BINS", "topological_regions"), arg_type="int")
    except configparser.NoSectionError:
        # In this case,
        return -1
    return len(topological_regions)


def _to_dataframe(performance: Dict[str, Dict[Any, List[float]]], metric: str) -> pandas.DataFrame:
    main_dict = dict()

    i = 0
    for channel_system, channel_system_performance in performance.items():
        for method, curve in channel_system_performance.items():
            for single_performance in curve:
                main_dict[f"row_{i}"] = {"Method": method, "Channel System": channel_system,
                                         f"Performance ({metric})": single_performance}
                i += 1

    return pandas.DataFrame.from_dict(main_dict, orient="index")


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


def main() -> None:
    results_folder = get_results_dir()

    # Define models to be used for plotting. See model_selection_rbp_ssi_zf for model selection
    models = {"Region Based Pooling": "5Fold_RBP_Shared_SharedRocketKernels_2023-06-30_11:14:26",
              "Interpolation": "5_Fold_FixedChannels_Interpolation_Inception_2023-06-30_05:48:13",
              "Zero-Filling": "5_Fold_FixedChannels_ZeroFill_Inception_2023-06-30_08:03:11"}

    # Select performance metrics
    metric_to_plot = "auc"

    # Loop through all experiments
    peak_performances: Dict[str, Dict[str, List[float]]] = dict()
    for model, path in models.items():
        model_path = os.path.join(results_folder, path)

        # Get folders name of all runs
        sub_folders = os.listdir(model_path)
        sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder[:4] == "Run_"]

        # Loop through all runs
        for run in sub_folders:
            # Get all metrics from the run
            histories = get_metrics_single_run(path=os.path.join(model_path, run))

            # Get performance on all channel systems
            for channel_system, history in histories.items():
                if channel_system[:5] != "test_":
                    continue

                # Compute peak of smoothed curve
                test_performance = history[metric_to_plot][0]

                # Store in main dict
                label = _ch_system_to_label(channel_system)
                if label in peak_performances:
                    if model in peak_performances[label]:
                        peak_performances[label][model].append(test_performance)
                    else:
                        peak_performances[label][model] = [test_performance]
                else:
                    peak_performances[label] = {model: [test_performance]}

    # Sort the dict (I know that this is a little too hard-coded, but I really have to fix this plot for the paper)
    data = {"Test (c=32)": peak_performances["Test (c=32)"],
            "Test (c=65)": peak_performances["Test (c=65)"],
            "Test (c=129)": peak_performances["Test (c=129)"]}

    # Convert to pandas dataframe
    data = _to_dataframe(performance=data, metric=metric_to_plot.upper())

    # --------------
    # Plotting
    # --------------
    # A few cosmetic settings
    font_size = 35
    rcParams["legend.fontsize"] = font_size
    rcParams["legend.title_fontsize"] = font_size

    # Plot
    ax = seaborn.violinplot(data=data, x="Channel System", y=f"Performance ({metric_to_plot.upper()})", hue="Method",
                            cut=0, bw=0.3)

    # Cosmetics
    ax.tick_params(labelsize=font_size, rotation=-0)
    ax.set_title(f"Sex Prediction with Varied Numbers of Channels", fontsize=font_size+5)
    ax.set_ylim((0.65, 1))
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)

    pyplot.show()


if __name__ == "__main__":
    main()
