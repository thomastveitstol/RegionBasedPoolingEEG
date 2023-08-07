import random

import numpy
import torch

from src.data.datasets.cleaned_child_data import CleanedChildChannelSystem, Reduced1CleanedChildChannelSystem, \
    Reduced3CleanedChildChannelSystem, CleanedChildData
from src.data.datasets.example_data import ExampleData
from src.data.line_separation.utils import project_head_shape
from src.models.modules.bins.regions_to_bins import _Bin, Regions2Bins, SharedPrecomputingRegions2Bins
from src.utils import ChGroup


# ------------------------
# _Bin class
# ------------------------
def test_bin_generate_module() -> None:
    """Test if the generate_module method of _Bin class runs and works properly"""
    # -------------------------
    # Define inputs
    # -------------------------
    # Channel systems to fit (after generating the splits)
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the split-algorithm on
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes = project_head_shape(nodes_3d)

    # -------------------------
    # Define model
    # -------------------------
    model = _Bin.generate_module(pooling_method="ContinuousAttention", channel_systems=channel_systems, nodes=nodes,
                                 min_nodes=2, candidate_region_splits=None, pooling_hyperparams={"cnn_units": 3})

    # -------------------------
    # Test some of the easy-to-check properties
    # -------------------------
    # Hyperparameter
    assert model.pooling_hyperparams["cnn_units"] == 3

    # Check that the path (stacking sequence) seems correct. Should be a tuple with int 'ChGroup' elements
    assert isinstance(model.path, tuple)
    assert all(isinstance(region, ChGroup) for region in model.path)

    # Check pooling method
    assert model.pooling_method == "ContinuousAttention"


def test_bin_forward() -> None:
    """Test if the forward method of the _Bin class is as expected"""
    random.seed(1)
    numpy.random.seed(2)

    # -------------------------
    # Define model
    # -------------------------
    # Channel systems to fit (after generating the splits)
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the split-algorithm on
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes = project_head_shape(nodes_3d)

    model_1 = _Bin.generate_module(pooling_method="ContinuousAttention", channel_systems=channel_systems, nodes=nodes,
                                   min_nodes=2, candidate_region_splits=None, pooling_hyperparams={"cnn_units": 3})
    model_2 = _Bin.generate_module(pooling_method="Mean", channel_systems=channel_systems, nodes=nodes,
                                   min_nodes=2, candidate_region_splits=None)

    # -------------------------
    # Generate data
    # -------------------------
    x = torch.rand(size=(10, 129, 300))

    # -------------------------
    # Forward method
    # -------------------------
    outputs_1 = model_1(x=x, channel_system_name=Reduced3CleanedChildChannelSystem().name,
                        channel_name_to_index=Reduced3CleanedChildChannelSystem().channel_name_to_index())
    outputs_2 = model_2(x=x, channel_system_name=Reduced3CleanedChildChannelSystem().name, precomputed=None,
                        channel_name_to_index=Reduced3CleanedChildChannelSystem().channel_name_to_index())

    assert outputs_1.size() == torch.Size([10, len(model_1.channel_split.allowed_ch_groups), 300])
    assert outputs_2.size() == torch.Size([10, len(model_2.channel_split.allowed_ch_groups), 300])


# ------------------------
# Regions2Bins class
# ------------------------
def test_regions_2_bins_generate_module() -> None:
    """Test if the generate_module method of Regions2Bins runs and works properly"""
    # -------------------------
    # Define inputs
    # -------------------------
    # Channel systems to fit (after generating the splits)
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the split-algorithm on
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes = project_head_shape(nodes_3d)

    # Which pooling methods to use (can vary from channel split to channel split, although not used in the paper)
    pooling_methods = ("ContinuousAttention", "Mean", "Mean", "ContinuousAttention")

    # Specify their hyperparameters
    pooling_hyperparams = ({"cnn_units": 7}, {}, {}, {"cnn_units": 3})

    # -------------------------
    # Define model
    # -------------------------
    model = Regions2Bins.generate_module(channel_systems=channel_systems,
                                         nodes=nodes,
                                         pooling_methods=pooling_methods,
                                         num_channel_splits=len(pooling_methods),
                                         min_nodes=2,
                                         pooling_hyperparams=pooling_hyperparams)

    # -------------------------
    # Test some of the easy-to-check properties
    # -------------------------
    # Check if the pooling methods are as specified
    assert model.pooling_methods == pooling_methods

    # Check if the pooling hyperparameters are as expected
    actual_hyperparams = model.pooling_hyperparams

    assert actual_hyperparams[0]["cnn_units"] == pooling_hyperparams[0]["cnn_units"]
    assert actual_hyperparams[-1]["cnn_units"] == pooling_hyperparams[-1]["cnn_units"]


def test_regions_2_bins_output_shape() -> None:
    """Test if the output shapes is as expected"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Define model
    # -------------------------
    # Channel systems to fit (after generating the splits)
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the split-algorithm on
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes = project_head_shape(nodes_3d)

    # Which pooling methods to use (can vary from channel split to channel split, although not used in the paper)
    pooling_methods = ("ContinuousAttention", "Mean", "Mean", "ContinuousAttention")

    # Specify their hyperparameters
    pooling_hyperparams = ({"cnn_units": 2, "max_kernel_size": 8, "depth": 2}, {}, {},
                           {"cnn_units": 2, "max_kernel_size": 8, "depth": 2})

    # Create model
    model = Regions2Bins.generate_module(channel_systems=channel_systems,
                                         nodes=nodes,
                                         pooling_methods=pooling_methods,
                                         num_channel_splits=len(pooling_methods),
                                         min_nodes=2,
                                         pooling_hyperparams=pooling_hyperparams).to(device)

    # -------------------------
    # Load data
    # -------------------------
    dataset = CleanedChildData()

    time_series_length = 300
    x = torch.tensor(dataset.load_eeg_data(subjects=dataset.get_subject_ids()[:1], truncate=time_series_length,
                                           time_series_start=0), dtype=torch.float).to(device)
    channel_system = dataset.channel_system

    # -------------------------
    # Pass data through the model
    # -------------------------
    # Forward pass
    outputs = model(x=x, channel_system_name=channel_system.name,
                    channel_name_to_index=channel_system.channel_name_to_index())

    # -------------------------
    # Test if output shape is as expected
    # -------------------------
    # Number of outputs equal the number of channel splits
    assert len(outputs) == len(pooling_methods)

    # Channel dimension of each channel split is the number of regions in the channel split
    assert tuple(output.size() for output in outputs) == \
           tuple(torch.Size([1, len(channel_split.allowed_ch_groups), 300]) for channel_split in model.channel_splits)


# ------------------------
# SharedPrecomputingRegions2Bins class
# ------------------------
def test_shared_rocket_generate_module() -> None:
    # -------------------------
    # Define inputs
    # -------------------------
    # Channel systems to fit (after generating the splits)
    channel_systems = (CleanedChildChannelSystem(), Reduced1CleanedChildChannelSystem(),
                       Reduced3CleanedChildChannelSystem())

    # Nodes/electrodes to fit the split-algorithm on
    nodes_3d = Reduced3CleanedChildChannelSystem().get_electrode_positions()
    nodes = project_head_shape(nodes_3d)

    # Specify hyperparameters
    num_kernels = 300
    max_receptive_field = 200

    # -------------------------
    # Define model
    # -------------------------
    model_1 = SharedPrecomputingRegions2Bins.generate_module(channel_systems=channel_systems,
                                                             nodes=nodes,
                                                             pooling_method="SharedRocketKernels",
                                                             num_channel_splits=4,
                                                             min_nodes=2,
                                                             num_kernels=num_kernels,
                                                             max_receptive_field=max_receptive_field,
                                                             rocket_implementation=3)
    model_2 = SharedPrecomputingRegions2Bins.generate_module(channel_systems=channel_systems,
                                                             nodes=nodes,
                                                             pooling_method="ConnectedSearch2",
                                                             num_channel_splits=9,
                                                             min_nodes=2,
                                                             num_kernels=num_kernels,
                                                             max_receptive_field=max_receptive_field,
                                                             rocket_implementation=3,
                                                             pooling_hyperparams={"latent_search_features": 32,
                                                                                  "share_edge_embeddings": True})

    # -------------------------
    # Test some of the easy-to-check properties
    # -------------------------
    # Check if the pooling methods are as specified
    assert model_1.pooling_methods == tuple(["SharedRocketKernels"] * 4)
    assert model_2.pooling_methods == tuple(["ConnectedSearch2"] * 9)

    # Check if the pooling hyperparameters are as expected
    actual_hyperparams_1 = model_1.pooling_hyperparams
    for actual, channel_split in zip(actual_hyperparams_1, model_1.channel_splits):
        assert actual["num_pooling_modules"] == len(channel_split.allowed_ch_groups)
        assert actual["in_features"] == num_kernels * 2
        assert actual["fc_block_id"] == 3
        assert actual["fc_units"] == [1]

    actual_hyperparams_2 = model_2.pooling_hyperparams
    for actual, channel_split in zip(actual_hyperparams_2, model_2.channel_splits):
        assert actual["num_regions"] == len(channel_split.allowed_ch_groups)
        assert actual["in_features"] == num_kernels * 2
        assert actual["latent_search_features"] == 32
        assert actual["share_edge_embeddings"]


def test_save_and_load() -> None:
    """Test if a loaded model gives the same output as the saved model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load dummy data
    # -------------------------
    dataset = ExampleData()

    x = torch.tensor(dataset.load_eeg_data(subjects=dataset.get_subject_ids()), dtype=torch.float).to(device)
    channel_system = dataset.channel_system

    # -------------------------
    # Define model
    # -------------------------
    group_pooling_module_hyperparams = {"num_kernels": 123, "rocket_implementation": 2}

    model = Regions2Bins.generate_module(pooling_methods="UnivariateRocket",
                                         channel_systems=channel_system, num_channel_splits=5, min_nodes=10,
                                         pooling_hyperparams=group_pooling_module_hyperparams,
                                         nodes=project_head_shape(dataset.channel_system.get_electrode_positions())
                                         ).to(device)

    # -------------------------
    # Pass through dummy data and save the model
    # -------------------------
    # Precompute
    precomputed = model.pre_compute_batch(x=x, channel_system_name=channel_system.name, to_cpu=False,
                                          channel_name_to_index=channel_system.channel_name_to_index(),
                                          batch_size=x.size()[0] // 7)

    # Forward pass
    with torch.no_grad():
        saved_outputs = model(x=x, channel_system_name=channel_system.name, precomputed=precomputed,
                              channel_name_to_index=channel_system.channel_name_to_index())

    # Save
    path = "/home/thomas/PycharmProjects/ChannelInvariance/test/models/modules/bins/regions_to_bins.pt"
    model.save(path=path)

    # -------------------------
    # Load model and run dummy data
    # through forward method
    # -------------------------
    model = Regions2Bins.from_disk(path=path).to(device)

    # Run the same dummy data through
    model.eval()
    with torch.no_grad():
        loaded_outputs = model(x=x, channel_system_name=channel_system.name,
                               channel_name_to_index=channel_system.channel_name_to_index(),
                               precomputed=precomputed)

    # -------------------------
    # Test if the outputs are equal
    # -------------------------
    # Length check
    assert len(saved_outputs) == len(loaded_outputs), f"Length of output tuples were expected to be equal, but were " \
                                                      f"{len(saved_outputs)} and {len(loaded_outputs)}"
    # Check all elements
    assert all(
        torch.equal(saved_output, loaded_output) for saved_output, loaded_output in zip(saved_outputs, loaded_outputs)
    ), f"The loaded model have a different output than the saved model"
