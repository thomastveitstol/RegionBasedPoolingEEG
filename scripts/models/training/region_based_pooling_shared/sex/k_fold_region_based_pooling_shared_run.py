"""
Script for running k-fold cross validation of region based pooling, with shared ROCKET kernels. That is, the ROCKET
features are pre-computed per channel, and shared across all regions and channel splits

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
import os
import argparse
import shutil
import random
import warnings
from configparser import ConfigParser
from datetime import date, datetime

import numpy
import torch
import torch.nn as nn
from torch import optim
from matplotlib import pyplot

from torch.utils.data import DataLoader

from src.config_functions import str_to_optional_type, str_to_list, to_dict, str_to_type
from src.data.data_generators.data_gen import NewStaticChannelPrecomputingDataGenerator
from src.data.line_separation.utils import project_head_shape
from src.data.paths import get_results_dir
from src.metrics import Histories, save_all_histories
from src.data.datasets.data_base import smallest_channel_system, EEGDataset
from src.models.nn_models.region_based_m1.region_based_m1_shared_precomputing_main_model import SharedRocketClassifierM1


def main() -> None:
    # Make reproducible ish
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # ----------------------------------------------
    # Hyperparameters - read and set
    # ----------------------------------------------
    # Read argparser and config file
    arg_parser = argparse.ArgumentParser(description="Run script of Main Model")
    arg_parser.add_argument("-c", "--config_path", type=str, help="Path to config (.ini) file")
    args = arg_parser.parse_args()

    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "config_files", args.config_path))

    # Regions to bins
    rocket_implementation = config.getint("REGIONS2BINS", "rocket_implementation")
    num_kernels = config.getint("REGIONS2BINS", "num_kernels")
    max_receptive_field = config.getint("REGIONS2BINS", "max_receptive_field")

    # Channel splits
    min_nodes = config.getint("CHANNELSPLITHYPERPARAMS", "min_nodes")
    num_channel_splits = config.getint("CHANNELSPLITHYPERPARAMS", "num_channel_splits")
    candidate_region_splits = tuple(str_to_list(config.get("CHANNELSPLITHYPERPARAMS", "candidate_region_splits",
                                                           fallback="[CentroidPolygons]"),
                                                arg_type="str"))

    # Pooling module hyperparameters
    pooling_module_hyperparams = to_dict(config["POOLINGMODULEHYPERPARAMS"])
    pooling_method = pooling_module_hyperparams.pop("name", "SharedRocketKernels")
    batch_norm = pooling_module_hyperparams.pop("batch_norm", False)

    # Training hyperparameters
    num_epochs = config.getint("TRAINING", "num_epochs")
    batch_size = config.getint("TRAINING", "batch_size")
    pre_computing_batch_size = config.getint("TRAINING", "pre_computing_batch_size", fallback=batch_size)
    learning_rate = config.getfloat("TRAINING", "learning_rate", fallback=0.001)

    # Data hyperparameters
    num_subjects = str_to_optional_type(config.get("DATA", "num_subjects"), arg_type="int")
    num_folds = config.getint("DATA", "num_folds")
    time_series_start = config.getint("DATA", "time_series_start", fallback=10000)
    sampling_freq = config.getfloat("DATA", "sampling_freq")
    num_seconds = config.getfloat("DATA", "num_seconds")
    seq_length = int(num_seconds * sampling_freq)

    # Stacked Bins Classifier
    stacked_bins_classifier_name = config.get("STACKEDBINSCLASSIFIER", "stacked_bins_classifier_name")
    stacked_bins_classifier_hyperparams = {"input_channels": None, "num_classes": 1, "time_steps": seq_length,
                                           "sampling_freq": sampling_freq}

    # Model
    model_name = f"{num_folds}Fold_RBP_Shared_{pooling_method}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------
    dataset: EEGDataset = str_to_type(config.get("DATA", "dataset"), arg_type="dataset")

    # Generate k folds for cross validation
    subjects = dataset.get_subject_ids()
    k_folds = dataset.k_fold_sex_split(subjects=tuple(subjects), num_folds=num_folds, num_subjects=num_subjects,
                                       force_balanced=True)  # Splitting the subject IDs
    subjects = [subject for fold in k_folds for subject in fold]

    # ----------------------------------------------
    # Load EEG Channel System Object
    # ----------------------------------------------
    train_channel_systems = tuple(str_to_list(config.get("DATA", "train_channel_systems"), arg_type="channel_system"))
    val_channel_systems = tuple(str_to_list(config.get("DATA", "val_channel_systems"), arg_type="channel_system"))
    allowed_channel_systems = tuple(str_to_list(config.get("DATA", "allowed_channel_systems"),
                                                arg_type="channel_system"))

    # Set channel system nodes to the smallest channel system
    nodes_3d = smallest_channel_system(allowed_channel_systems).get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # Add train_channel_systems and val_channel_systems to allowed_channel_systems
    allowed_channel_systems = tuple(set(train_channel_systems + val_channel_systems + allowed_channel_systems))

    # ----------------------------------------------
    # Load EEG data
    # ----------------------------------------------
    all_eeg = torch.tensor(dataset.load_eeg_data(subjects=subjects, truncate=seq_length,
                                                 time_series_start=time_series_start), dtype=torch.float32)
    all_targets = torch.tensor(dataset.load_targets(target="sex", subjects=subjects), dtype=torch.float32)

    # ----------------------------------------------
    # Create folder
    # ----------------------------------------------
    parent_dir = get_results_dir()
    path = os.path.join(parent_dir, f"{model_name}_{date.today()}_{datetime.now().strftime('%H:%M:%S')}")
    os.mkdir(path)

    # Save config file
    shutil.copy(src=os.path.join(os.path.dirname(__file__), "config_files", args.config_path),
                dst=os.path.join(path, args.config_path.split("/")[-1]))

    # Suppress RuntimeWarning, they occur often when computing metrics
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for fold in range(num_folds):

        print("\n\n")
        print("-----------------------------\n")
        print(f"Fold {fold + 1}/{num_folds}")
        print("\n-----------------------------")
        print("\n\n")

        # ----------------------------------------------
        # Data generators and data loaders
        # ----------------------------------------------
        # Train generator and loader
        train_subjects = tuple(k_fold for i, k_fold in enumerate(k_folds) if i != fold)
        val_subjects = train_subjects[-1]  # Select validation set
        train_subjects = train_subjects[:-1]  # Remove validation set
        train_subjects = tuple(item for tuple_item in train_subjects for item in tuple_item)  # unpack tuple
        train_indices = tuple(subjects.index(subject) for subject in train_subjects)

        train_gen = NewStaticChannelPrecomputingDataGenerator(x=all_eeg[train_indices, ...],
                                                              y=all_targets[train_indices, ...])
        train_loader = DataLoader(dataset=train_gen, batch_size=batch_size, shuffle=True)

        # Validation generator and loader
        val_indices = tuple(subjects.index(subject) for subject in val_subjects)

        val_gen = NewStaticChannelPrecomputingDataGenerator(x=all_eeg[val_indices, ...],
                                                            y=all_targets[val_indices, ...])
        val_loader = DataLoader(dataset=val_gen, batch_size=batch_size, shuffle=True)

        # Test generator and loader
        test_subjects = k_folds[fold]
        test_indices = tuple(subjects.index(subject) for subject in test_subjects)

        test_gen = NewStaticChannelPrecomputingDataGenerator(x=all_eeg[test_indices, ...],
                                                             y=all_targets[test_indices, ...])
        test_loader = DataLoader(dataset=test_gen, batch_size=batch_size, shuffle=True)

        assert len(set(train_indices)) == len(train_indices) and len(set(val_indices)) == len(val_indices) and \
               len(set(test_indices)) == len(test_indices)
        assert not set(train_indices) & set(val_indices), "Data leakage: Found subjects in both train and val dataset"
        assert not set(train_indices) & set(test_indices), "Data leakage: Found subjects in both train and test dataset"
        assert not set(val_indices) & set(test_indices), "Data leakage: Found subjects in both val and test dataset"
        # ----------------------------------------------
        # Define model, optimiser, loss and Histories objects
        # ----------------------------------------------
        print("Defining model, loss, optimiser, and Histories objects...")

        # Model
        model = SharedRocketClassifierM1.generate_model(
            stacked_bins_classifier_name=stacked_bins_classifier_name,
            stacked_bins_classifier_hyperparams=stacked_bins_classifier_hyperparams,
            nodes=nodes,
            num_channel_splits=num_channel_splits,
            min_nodes=min_nodes,
            candidate_region_splits=candidate_region_splits,
            pooling_method=pooling_method,
            pooling_hyperparams=pooling_module_hyperparams,
            channel_systems=allowed_channel_systems,
            normalise_region_representations=True,
            rocket_implementation=rocket_implementation,
            num_kernels=num_kernels,
            max_receptive_field=max_receptive_field,
            batch_norm=batch_norm).to(device)

        # Loss
        criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # Optimiser
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)

        # Histories
        train_history = Histories()
        val_histories = {channel_system.name: Histories() for channel_system in val_channel_systems}
        test_histories = {channel_system.name: Histories() for channel_system in val_channel_systems}

        # -------------------------------
        # Pre-compute features
        # -------------------------------
        if model.supports_precomputing:
            print("\nPre-computing features (this may take a while)...")
            with torch.no_grad():
                # Precompute for training
                print("\tTraining dataset:")
                train_loader.dataset.pre_computed_features = model.pre_compute(train_gen.x.to(device),
                                                                               batch_size=pre_computing_batch_size)

                # Pre-compute for validation
                print("\tValidation dataset:")
                val_loader.dataset.pre_computed_features = model.pre_compute(val_gen.x.to(device),
                                                                             batch_size=pre_computing_batch_size)

        # ----------------------------------------------
        # Fit model
        # ----------------------------------------------
        print("Fitting model...")
        model.fit_model(train_loader=train_loader, val_loader=val_loader, train_history=train_history,
                        val_histories=val_histories, train_channel_systems=train_channel_systems,
                        val_channel_systems=val_channel_systems, num_epochs=num_epochs, device=device,
                        criterion=criterion, optimiser=optimiser, activation_function=torch.sigmoid)

        # ----------------------------------------------
        # Test model
        # ----------------------------------------------
        # Precompute
        if model.supports_precomputing:
            print("\nPre-computing features for test set (this may take a while)...")
            with torch.no_grad():
                # Clear memory first
                train_loader.dataset.pre_computed_features = None
                val_loader.dataset.pre_computed_features = None

                # Precompute the test set
                test_loader.dataset.pre_computed_features = model.pre_compute(test_gen.x.to(device),
                                                                              batch_size=pre_computing_batch_size)

        # Test
        print("Testing model on test set...")
        model.eval()
        model.test_model(loader=test_loader, channel_systems=val_channel_systems, histories=test_histories,
                         device=device, activation_function=torch.sigmoid)

        # ----------------------------------------------
        # Save
        # ----------------------------------------------
        # Save histories
        sub_folder = f"Run_{fold}"
        os.mkdir(os.path.join(path, sub_folder))
        pyplot.close("all")

        validation_histories = {f"val_{name}": history for name, history in val_histories.items()}
        testing_histories = {f"test_{name}": history for name, history in test_histories.items()}
        histories = {**validation_histories, **testing_histories, f"train_All": train_history}

        save_all_histories(histories=histories, path=os.path.join(path, sub_folder))

        # Save model
        model.save(path=os.path.join(path, sub_folder))


if __name__ == "__main__":
    main()
