import os
from typing import Any

import numpy
import torch
import torch.nn as nn
from torch import optim

from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem, Reduced1CleanedChildChannelSystem, \
    Reduced3CleanedChildChannelSystem, CleanedChildData
from src.data.datasets.data_base import BaseChannelSystem
from src.data.line_separation.utils import project_head_shape
from src.models.nn_models.region_based_m1.region_based_m1_main_model import Regions2BinsClassifierM1


def _train_and_save(path: str, batch: int, channels: int, time_steps: int, test_x: torch.Tensor,
                    channel_system: BaseChannelSystem, model: Regions2BinsClassifierM1, device: torch.device,
                    test_pre_computed: Any, precompute: bool = False) -> torch.Tensor:
    # Model, loss and optimizer
    criterion = nn.MSELoss(reduction="mean")
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    channel_system_name = channel_system.name
    channel_name_to_index = channel_system.channel_name_to_index()

    # -------------------------
    # Train model
    # -------------------------
    if precompute:
        # Pre-compute dummy ROCKET-like features
        rocket_like_features = model.pre_compute(torch.rand(size=(batch, channels, time_steps)).to(device),
                                                 channel_system_name=channel_system_name, to_cpu=False,
                                                 channel_name_to_index=channel_name_to_index, batch_size=batch // 2)
    else:
        rocket_like_features = None

    for _ in range(20):
        # Generating input and output dummy data. Setting batch-size to the complete dataset
        x = torch.rand(size=(batch, channels, time_steps)).to(device)
        y = torch.zeros(size=(batch, 1)).to(device)

        # Forward pass
        scores = model(x, precomputed=rocket_like_features, channel_system_name=channel_system_name,
                       channel_name_to_index=channel_name_to_index)

        # Compute loss
        loss = criterion(scores, y)

        # Do a training step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # ---------------------------
    # Pass test_x through the model
    # ---------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(test_x, channel_system_name=channel_system_name, channel_name_to_index=channel_name_to_index,
                        precomputed=test_pre_computed)

    # ---------------------------
    # Save model
    # ---------------------------
    model.save(path=path)

    return outputs


def test_save_and_load() -> None:
    """Test if a loaded model gives the same results as the saved model"""
    # Define path for saving the model
    path = os.path.dirname(__file__)

    # ------------------
    # Define models
    # ------------------
    # Hyperparameters
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # Create model
    device = torch.device("cpu")
    model_1 = Regions2BinsClassifierM1.generate_model(num_channel_splits=2, min_nodes=2, nodes=nodes,
                                                      pooling_methods="ContinuousAttention",
                                                      pooling_hyperparams={"max_kernel_size": 10, "depth": 1,
                                                                           "cnn_units": 1},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems).to(device)
    model_2 = Regions2BinsClassifierM1.generate_model(num_channel_splits=4, min_nodes=3, nodes=nodes,
                                                      pooling_methods="UnivariateRocket",
                                                      pooling_hyperparams={"num_kernels": 30,
                                                                           "rocket_implementation": 2,
                                                                           "max_receptive_field": 12},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 7},
                                                      channel_systems=channel_systems).to(device)
    model_3 = Regions2BinsClassifierM1.generate_model(num_channel_splits=31, min_nodes=3, nodes=nodes,
                                                      pooling_methods="Mean",
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 4},
                                                      channel_systems=channel_systems).to(device)

    # ------------------
    # Load/create data
    # ------------------
    dataset = CleanedChildData()

    x = torch.rand(size=(4, dataset.num_channels, 300)).to(device)
    channel_system = dataset.channel_system

    # Univariate ROCKET needs precomputing
    pre_computed_2 = model_2.pre_compute(x=x, channel_system_name=channel_system.name, batch_size=3, to_cpu=False,
                                         channel_name_to_index=channel_system.channel_name_to_index())

    # -------------------------
    # Train and save the model. Also,
    # get the outputs on dummy data
    # -------------------------
    path_1 = os.path.join(path, "mean")
    path_2 = os.path.join(path, "continuous")
    path_3 = os.path.join(path, "univariate_rocket")

    saved_outputs_1 = _train_and_save(path=path_1, batch=3, channels=dataset.num_channels, time_steps=300, test_x=x,
                                      test_pre_computed=None, channel_system=channel_system, model=model_1,
                                      device=device)
    saved_outputs_2 = _train_and_save(path=path_2, batch=3, channels=dataset.num_channels, time_steps=300, test_x=x,
                                      test_pre_computed=pre_computed_2, channel_system=channel_system, model=model_2,
                                      device=device, precompute=True)
    saved_outputs_3 = _train_and_save(path=path_3, batch=3, channels=dataset.num_channels, time_steps=300, test_x=x,
                                      test_pre_computed=None, channel_system=channel_system, model=model_3,
                                      device=device)

    # -------------------------
    # Load models and run the same data
    # through forward method
    # -------------------------
    # Load models
    model_1_loaded = Regions2BinsClassifierM1.from_disk(path=path_1).to(device)
    model_2_loaded = Regions2BinsClassifierM1.from_disk(path=path_2).to(device)
    model_3_loaded = Regions2BinsClassifierM1.from_disk(path=path_3).to(device)

    # Forward pass
    model_1_loaded.eval()
    model_2_loaded.eval()
    model_3_loaded.eval()

    with torch.no_grad():
        loaded_outputs_1 = model_1_loaded(x=x, channel_system_name=channel_system.name,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          precomputed=None)
        loaded_outputs_2 = model_2_loaded(x=x, channel_system_name=channel_system.name,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          precomputed=pre_computed_2)
        loaded_outputs_3 = model_3_loaded(x=x, channel_system_name=channel_system.name,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          precomputed=None)

    # -------------------------
    # Test if the outputs are equal
    # -------------------------
    assert torch.equal(saved_outputs_1, loaded_outputs_1)
    assert torch.equal(saved_outputs_2, loaded_outputs_2)
    assert torch.equal(saved_outputs_3, loaded_outputs_3)

    assert not torch.equal(saved_outputs_1, loaded_outputs_2)
    assert not torch.equal(saved_outputs_2, loaded_outputs_3)
    assert not torch.equal(saved_outputs_2, loaded_outputs_3)


def test_generate_model() -> None:
    """Test if the generate model method works properly (that it runs, and property checks)"""
    # ------------------
    # Define model inputs
    # ------------------
    # Hyperparameters
    num_channel_splits = 4
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # ------------------
    # Define model
    # ------------------
    model = Regions2BinsClassifierM1.generate_model(num_channel_splits=num_channel_splits, min_nodes=2, nodes=nodes,
                                                    pooling_methods=("Mean", "ContinuousAttention",
                                                                     "UnivariateRocketGroup"),
                                                    stacked_bins_classifier_name="Inception",
                                                    stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                    channel_systems=channel_systems)

    # ------------------
    # Test some properties
    # ------------------
    # Test number of channel splits
    assert len(model.channel_splits) == num_channel_splits

    # Test if the pooling modules are correct (remember that the 'pooling_methods' argument is cycled)
    assert model._regions_2_bins_module.pooling_methods == ("Mean", "ContinuousAttention", "UnivariateRocketGroup",
                                                            "Mean")


def test_output_shape() -> None:
    """Test if the output shape is as expected after running forward method"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------
    # Define model
    # ------------------
    # Hyperparameters
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # Create model
    model_1 = Regions2BinsClassifierM1.generate_model(num_channel_splits=3, min_nodes=2, nodes=nodes,
                                                      pooling_methods="ContinuousAttention",
                                                      pooling_hyperparams={"max_kernel_size": 10, "depth": 2,
                                                                           "cnn_units": 2},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems).to(device)
    model_2 = Regions2BinsClassifierM1.generate_model(num_channel_splits=9, min_nodes=1, nodes=nodes,
                                                      pooling_methods="UnivariateRocket",
                                                      pooling_hyperparams={"num_kernels": 30,
                                                                           "max_receptive_field": 12},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 7},
                                                      channel_systems=channel_systems).to(device)
    model_3 = Regions2BinsClassifierM1.generate_model(num_channel_splits=31, min_nodes=3, nodes=nodes,
                                                      pooling_methods="Mean",
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 4},
                                                      channel_systems=channel_systems).to(device)

    # ------------------
    # Load data
    # ------------------
    dataset = CleanedChildData()

    x = torch.tensor(dataset.load_eeg_data(subjects=dataset.get_subject_ids()[:4], truncate=2000),
                     dtype=torch.float32).to(device)

    # ------------------
    # Forward method
    # ------------------
    # ContinuousAttention
    outputs_1 = model_1(x=x, channel_system_name=dataset.channel_system.name,
                        channel_name_to_index=dataset.channel_system.channel_name_to_index())

    # UnivariateRocket (requires precomputing)
    pre_computed_2 = model_2.pre_compute(x=x, channel_system_name=dataset.channel_system.name, batch_size=3,
                                         to_cpu=False,
                                         channel_name_to_index=dataset.channel_system.channel_name_to_index())
    outputs_2 = model_2(x=x, channel_system_name=dataset.channel_system.name, precomputed=pre_computed_2,
                        channel_name_to_index=dataset.channel_system.channel_name_to_index())

    # Mean
    outputs_3 = model_3(x=x, channel_system_name=dataset.channel_system.name,
                        channel_name_to_index=dataset.channel_system.channel_name_to_index())

    # ------------------
    # Test output shape
    # ------------------
    # Should be (num_subjects, num_classes)
    assert outputs_1.size() == torch.Size([4, 3])
    assert outputs_2.size() == torch.Size([4, 7])
    assert outputs_3.size() == torch.Size([4, 4])
