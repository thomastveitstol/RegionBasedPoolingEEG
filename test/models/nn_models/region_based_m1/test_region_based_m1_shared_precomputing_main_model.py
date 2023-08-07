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
from src.models.nn_models.region_based_m1.region_based_m1_shared_precomputing_main_model import SharedRocketClassifierM1


def _train_and_save(path: str, batch: int, channels: int, time_steps: int, test_x: torch.Tensor,
                    channel_system: BaseChannelSystem, model: SharedRocketClassifierM1, device: torch.device,
                    test_pre_computed: Any, precompute: bool = True) -> torch.Tensor:
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
                                                 batch_size=batch // 2).to(device)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # ------------------
    # Define models
    # ------------------
    model_1 = SharedRocketClassifierM1.generate_model(num_channel_splits=4, num_kernels=101, rocket_implementation=2,
                                                      max_receptive_field=200, pooling_method="SharedRocketKernels",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems).to(device)
    model_2 = SharedRocketClassifierM1.generate_model(num_channel_splits=6, num_kernels=203, rocket_implementation=2,
                                                      max_receptive_field=200, pooling_method="ConnectedSearch2",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      pooling_hyperparams={"latent_search_features": 44,
                                                                           "share_edge_embeddings": True},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 7},
                                                      channel_systems=channel_systems).to(device)

    # ------------------
    # Load data
    # ------------------
    dataset = CleanedChildData()

    x = torch.rand(size=(4, dataset.num_channels, 2001)).to(device)
    channel_system = dataset.channel_system

    # Precompute features
    precomputed_1 = model_1.pre_compute(x, batch_size=3).to(device)
    precomputed_2 = model_2.pre_compute(x, batch_size=3).to(device)

    # -------------------------
    # Train and save the model. Also, get the
    # outputs on the data
    # -------------------------
    path_1 = os.path.join(path, "shared_rocket")
    path_2 = os.path.join(path, "connected")

    saved_outputs_1 = _train_and_save(path=path_1, batch=3, channels=dataset.num_channels, time_steps=2001, test_x=x,
                                      test_pre_computed=precomputed_1, channel_system=channel_system, model=model_1,
                                      device=device)
    saved_outputs_2 = _train_and_save(path=path_2, batch=3, channels=dataset.num_channels, time_steps=2001, test_x=x,
                                      test_pre_computed=precomputed_2, channel_system=channel_system, model=model_2,
                                      device=device, precompute=True)

    # -------------------------
    # Load models and run the same data
    # through forward method
    # -------------------------
    # Load models
    model_1_loaded = SharedRocketClassifierM1.from_disk(path=path_1).to(device)
    model_2_loaded = SharedRocketClassifierM1.from_disk(path=path_2).to(device)

    # Forward pass
    model_1_loaded.eval()
    model_2_loaded.eval()

    with torch.no_grad():
        loaded_outputs_1 = model_1_loaded(x=x, channel_system_name=channel_system.name,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          precomputed=precomputed_1)
        loaded_outputs_2 = model_2_loaded(x=x, channel_system_name=channel_system.name,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          precomputed=precomputed_2)

    # -------------------------
    # Test if the outputs are equal
    # -------------------------
    assert torch.equal(saved_outputs_1, loaded_outputs_1)
    assert torch.equal(saved_outputs_2, loaded_outputs_2)

    assert not torch.equal(saved_outputs_1, loaded_outputs_2)


def test_generate_model() -> None:
    """Test if the generate model method works properly (that it runs, and property checks)"""
    # ------------------
    # Define model inputs
    # ------------------
    # Hyperparameters
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # ------------------
    # Define models
    # ------------------
    model_1 = SharedRocketClassifierM1.generate_model(num_channel_splits=4, num_kernels=101, rocket_implementation=3,
                                                      max_receptive_field=200, pooling_method="SharedRocketKernels",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems)
    model_2 = SharedRocketClassifierM1.generate_model(num_channel_splits=6, num_kernels=101, rocket_implementation=3,
                                                      max_receptive_field=200, pooling_method="ConnectedSearch2",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      pooling_hyperparams={"latent_search_features": 44,
                                                                           "share_edge_embeddings": True},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems)

    # ------------------
    # Test some properties
    # ------------------
    # Test number of channel splits
    assert len(model_1.channel_splits) == 4
    assert len(model_2.channel_splits) == 6

    # Test if the pooling modules are correct (remember that the 'pooling_method' argument is cycled)
    assert model_1._regions_2_bins_module.pooling_methods == tuple(["SharedRocketKernels"] * 4)
    assert model_2._regions_2_bins_module.pooling_methods == tuple(["ConnectedSearch2"] * 6)


def test_output_shape() -> None:
    """Test if the output shape is as expected after running forward method"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the regions
    nodes_3d = CleanedChildChannelSystem().get_electrode_positions()
    nodes_2d = project_head_shape(nodes_3d)
    points = numpy.array(tuple(nodes_2d.values()))
    nodes = {f'Ch_{i_}': point for i_, point in enumerate(points)}

    # ------------------
    # Define models
    # ------------------
    model_1 = SharedRocketClassifierM1.generate_model(num_channel_splits=4, num_kernels=101, rocket_implementation=3,
                                                      max_receptive_field=200, pooling_method="SharedRocketKernels",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 3},
                                                      channel_systems=channel_systems).to(device)
    model_2 = SharedRocketClassifierM1.generate_model(num_channel_splits=6, num_kernels=203, rocket_implementation=3,
                                                      max_receptive_field=200, pooling_method="ConnectedSearch2",
                                                      min_nodes=2, nodes=nodes, candidate_region_splits=None,
                                                      pooling_hyperparams={"latent_search_features": 44,
                                                                           "share_edge_embeddings": True},
                                                      stacked_bins_classifier_name="Inception",
                                                      stacked_bins_classifier_hyperparams={"num_classes": 7},
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
    # Precomputing
    precomputed_1 = model_1.pre_compute(x, batch_size=3).to(device)
    precomputed_2 = model_2.pre_compute(x, batch_size=3).to(device)

    assert precomputed_1.size() == torch.Size([4, 129, 202])
    assert precomputed_2.size() == torch.Size([4, 129, 406])

    # Forward pass
    outputs_1 = model_1(x=x, precomputed=precomputed_1, channel_system_name=dataset.name,
                        channel_name_to_index=dataset.channel_system.channel_name_to_index())
    outputs_2 = model_2(x=x, precomputed=precomputed_2, channel_system_name=dataset.name,
                        channel_name_to_index=dataset.channel_system.channel_name_to_index())

    assert outputs_1.size() == torch.Size([4, 3])
    assert outputs_2.size() == torch.Size([4, 7])
