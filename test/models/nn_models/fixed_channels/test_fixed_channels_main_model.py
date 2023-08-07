import os

import torch
import torch.nn as nn
from torch import optim

from src.models.nn_models.fixed_channels.fixed_channels_main_model import FixedChannelDimMainModel


def _train_and_save(path: str, classifier_name: str, batch: int, channels: int, time_steps: int,
                    test_x: torch.Tensor) -> torch.Tensor:
    """
    Train and save a model. Also, return the output of the saved model when 'test_x' is passed through forward method
    Args:
        path: path to save the model
        classifier_name: classifier name, e.g. 'Inception'
        batch: batch size to use
        channels: number of channels to use
        time_steps: number of time steps to use
        test_x: the tensor to make predictions on

    Returns:
        the model output with 'test_x' as input: model(test_x)
    """
    # Model, loss and optimizer
    model = FixedChannelDimMainModel(classifier_name=classifier_name, in_channels=channels, num_classes=1)

    criterion = nn.MSELoss(reduction="mean")
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------
    # Train model
    #
    # The model will be trained to output zeros
    # ---------------------------
    for _ in range(20):
        # Generating input and output dummy data. Setting batch-size to the complete dataset
        x = torch.rand(size=(batch, channels, time_steps))
        y = torch.zeros(size=(batch, 1))

        # Forward pass
        scores = model(x)

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
        outputs = model(test_x)

    # ---------------------------
    # Save model
    # ---------------------------
    model.save(path=path)

    return outputs


def test_save_and_load() -> None:
    """Test if a loaded model gives the same results as the saved model"""
    # path for saving
    path = os.path.dirname(__file__)

    # Hyperparameters
    classifier_name = "Inception"
    batch, channels, time_steps = 4, 11, 300  # For the dummy data

    # Dummy data. This will be passed through the model before saving and after loading, to see if the outputs are equal
    x = torch.rand(size=(3, channels, time_steps))

    # -------------------------
    # Train and save the model. Also,
    # get the outputs on dummy data
    # -------------------------
    saved_outputs = _train_and_save(path=path, batch=batch, channels=channels, time_steps=time_steps, test_x=x,
                                    classifier_name=classifier_name)

    # -------------------------
    # Load model and run dummy data
    # through forward method
    # -------------------------
    model = FixedChannelDimMainModel.from_disk(path=path)

    # Run the same dummy data through
    model.eval()
    with torch.no_grad():
        loaded_outputs = model(x)

    # -------------------------
    # Test if the outputs are equal
    # -------------------------
    assert torch.equal(saved_outputs, loaded_outputs)
