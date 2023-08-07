"""
Script for running k-fold cross validation using models with fixed input dimension, but removing channels by
interpolation
"""
import os
import argparse
import random
import shutil
from configparser import ConfigParser
from datetime import date, datetime
import warnings

import numpy
import torch
from torch import optim
from matplotlib import pyplot
from torch.utils.data import DataLoader

from src.data.paths import get_results_dir
from src.metrics import Histories, save_all_histories
from src.config_functions import str_to_optional_type, str_to_list, str_to_type
from src.data.data_generators.data_gen import PlainDataGenerator, InterpolatingDataGenerator, \
    batch_interpolate_bad_channels, zero_fill_input
from src.data.datasets.data_base import get_illegal_channels, channel_names_to_indices, EEGDataset
from src.models.modules.classifiers.utils import get_loss
from src.models.nn_models.fixed_channels.fixed_channels_main_model import FixedChannelDimMainModel


def main() -> None:
    # Make reproducible ish
    meaning_of_life = 42

    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    # ----------------------------------------------
    # Hyperparameters - read and set
    # ----------------------------------------------
    # Read argparser and config file
    arg_parser = argparse.ArgumentParser(description="Run script of Main Model (fixed channel input)")
    arg_parser.add_argument("-c", "--config_path", type=str, help="Path to config (.ini) file")
    args = arg_parser.parse_args()

    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "config_files", args.config_path))

    # Set all hyperparameters
    classifier_name = config.get("MODEL", "classifier_name")
    interpolation_method = config.get("MODEL", "interpolation_method", fallback="spline")
    model_name = f"FixedChannels_Interpolation_{classifier_name}"
    num_classes = config.getint("MODEL", "num_classes", fallback=1)

    # Data hyperparameters
    num_subjects = str_to_optional_type(config.get("DATA", "num_subjects"), arg_type="int")
    num_folds = config.getint("DATA", "num_folds")
    sampling_freq = config.getfloat("DATA", "sampling_freq")
    num_seconds = config.getfloat("DATA", "num_seconds")
    seq_length = int(num_seconds * sampling_freq)

    # Training hyperparameters
    num_epochs = config.getint("TRAINING", "num_epochs")
    batch_size = config.getint("TRAINING", "batch_size")
    learning_rate = config.getfloat("TRAINING", "learning_rate", fallback=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------
    dataset: EEGDataset = str_to_type(config.get("DATA", "dataset"), arg_type="dataset")

    # Generate k folds for cross validation
    subjects = dataset.get_subject_ids()
    k_folds = dataset.k_fold_sex_split(subjects=tuple(subjects), num_folds=num_folds, num_subjects=num_subjects,
                                       force_balanced=True)  # Splitting the subject IDs

    # ----------------------------------------------
    # Load EEG Channel System Object
    # ----------------------------------------------
    train_channel_system = dataset.channel_system
    channel_name_to_index = dataset.channel_system.channel_name_to_index()

    val_channel_systems = tuple(str_to_list(config.get("DATA", "val_channel_systems"), arg_type="channel_system"))

    num_val_channel_systems = len(val_channel_systems)

    illegal_channels = get_illegal_channels(main_channel_system=train_channel_system,
                                            reduced_channel_systems=val_channel_systems+(train_channel_system,))

    # ----------------------------------------------
    # Create folder
    # ----------------------------------------------
    parent_dir = get_results_dir()
    path = os.path.join(parent_dir, f"{num_folds}_Fold_{model_name}_{date.today()}_"
                                    f"{datetime.now().strftime('%H:%M:%S')}")
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
        # Data generators and data loaders  todo: very inefficient
        # ----------------------------------------------
        # Train generator and loader
        train_subjects = tuple(k_fold for i, k_fold in enumerate(k_folds) if i != fold)
        val_subjects = train_subjects[-1]  # Select validation set
        train_subjects = train_subjects[:-1]  # Remove validation set
        train_subjects = tuple(item for tuple_item in train_subjects for item in tuple_item)  # unpack tuple

        train_gen = PlainDataGenerator(subjects=list(train_subjects), dataset=dataset, seq_length=seq_length,
                                       time_series_start=10000, target="sex")
        train_loader = DataLoader(dataset=train_gen, batch_size=batch_size, shuffle=True)

        # Validation generator and loader
        val_gen = InterpolatingDataGenerator(subjects=val_subjects,
                                             dataset=dataset,
                                             seq_length=seq_length,
                                             time_series_start=10000, target="sex")
        val_loader = DataLoader(dataset=val_gen, batch_size=batch_size, shuffle=True)

        # Test generator and loader
        test_subjects = k_folds[fold]
        test_gen = InterpolatingDataGenerator(subjects=test_subjects,
                                              dataset=dataset,
                                              seq_length=seq_length,
                                              time_series_start=10000, target="sex")
        test_loader = DataLoader(dataset=test_gen, batch_size=batch_size, shuffle=True)

        assert len(set(train_subjects)) == len(train_subjects) and len(set(val_subjects)) == len(val_subjects) and \
               len(set(test_subjects)) == len(test_subjects)
        assert not set(train_subjects) & set(val_subjects), "Data leakage: Found subjects in both train and val dataset"
        assert not set(train_subjects) & set(test_subjects), "Data leakage: Found subjects in both train and test " \
                                                             "dataset"
        assert not set(val_subjects) & set(test_subjects), "Data leakage: Found subjects in both val and test dataset"

        # Do interpolation
        print("Interpolating...")
        for i, (channel_system_name, removed_channels) in enumerate(illegal_channels.items()):
            print(f"\t({i + 1}/{num_val_channel_systems+1})\t Interpolating: {channel_system_name}")
            # Zero-fill first to be safe
            channel_indices = channel_names_to_indices(channel_names=removed_channels,
                                                       channel_name_to_index=channel_name_to_index)
            zero_filled_x = zero_fill_input(val_gen.x, channel_indices=channel_indices).numpy()

            # Interpolate and add to val generator
            val_gen.x_interpolated[channel_system_name] = torch.tensor(
                data=batch_interpolate_bad_channels(
                    x=zero_filled_x, all_channel_names=list(channel_name_to_index.keys()), sampling_freq=sampling_freq,
                    bad_channels=removed_channels, method=interpolation_method),
                dtype=torch.float)  # .to(device)

        # ----------------------------------------------
        # Define model, optimiser, loss, and Histories objects
        # ----------------------------------------------
        print("Defining model, loss, optimiser, and Histories objects...")
        hyperparameters = {"in_channels": train_channel_system.num_channels, "num_classes": num_classes,
                           "time_steps": seq_length, "sampling_freq": sampling_freq}

        # Model
        model = FixedChannelDimMainModel(classifier_name=classifier_name, **hyperparameters).to(device)

        # Loss
        criterion = get_loss(activation_function=model.final_activation,
                             num_classes=hyperparameters["num_classes"])(reduction="mean")

        # Optimiser
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)

        # Histories
        train_history = Histories()
        val_histories = {channel_system.name: Histories()
                         for channel_system in val_channel_systems + (train_channel_system,)}
        test_histories = {channel_system.name: Histories()
                          for channel_system in val_channel_systems + (train_channel_system,)}

        # ------------------
        # Fit model
        # ------------------
        train_history, val_histories = model.fit_model(train_loader=train_loader, val_loader=val_loader,
                                                       train_history=train_history, val_histories=val_histories,
                                                       num_epochs=num_epochs, device=device, criterion=criterion,
                                                       optimiser=optimiser, activation_function=torch.sigmoid)

        # ------------------
        # Test model
        # ------------------
        # Do interpolation
        print("Interpolating test data...")
        val_gen.clear_memory()  # Clear memory
        for i, (channel_system_name, removed_channels) in enumerate(illegal_channels.items()):
            print(f"\t({i + 1}/{num_val_channel_systems+1})\t Interpolating: {channel_system_name}")
            # Zero-fill first to be safe
            channel_indices = channel_names_to_indices(channel_names=removed_channels,
                                                       channel_name_to_index=channel_name_to_index)
            zero_filled_x = zero_fill_input(test_gen.x, channel_indices=channel_indices).numpy()

            # Interpolate and add to test generator
            test_gen.x_interpolated[channel_system_name] = torch.tensor(
                data=batch_interpolate_bad_channels(
                    x=zero_filled_x, all_channel_names=list(channel_name_to_index.keys()), sampling_freq=sampling_freq,
                    bad_channels=removed_channels, method=interpolation_method),
                dtype=torch.float)

        # Test
        print("Testing model on test set...")
        model.eval()
        model.test_model(loader=test_loader, histories=test_histories, device=device, activation_function=torch.sigmoid)

        # ----------------------------------------------
        # Save all figures
        # ----------------------------------------------
        sub_folder = f"Run_{fold}"
        os.mkdir(os.path.join(path, sub_folder))
        pyplot.close("all")

        # Save histories
        validation_histories = {f"val_{name}": history for name, history in val_histories.items()}
        testing_histories = {f"test_{name}": history for name, history in test_histories.items()}
        histories = {**validation_histories, **testing_histories, f"train_{train_channel_system.name}": train_history}
        save_all_histories(histories=histories, path=os.path.join(path, sub_folder))

        # Save last model
        model.save(path=os.path.join(path, sub_folder))


if __name__ == "__main__":
    main()
