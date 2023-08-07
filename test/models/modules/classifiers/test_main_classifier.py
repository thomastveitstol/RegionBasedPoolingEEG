import torch

from src.models.modules.classifiers.main_mts_classifier import MTSClassifier


def test_expected_output_size() -> None:
    """Test if all implemented models run with an input with shape=(batch, channels, time_steps), with the expected
    output shape"""
    # ---------------------------
    # Make dummy data (with dummy hyperparams)
    # ---------------------------
    dummy_batch, dummy_channels, dummy_time_steps, dummy_sampling_freq = 10, 32, 2000, 128
    dummy_data = torch.rand(size=(dummy_batch, dummy_channels, dummy_time_steps))

    dummy_output_dim = 7
    expected_output_size = torch.Size([dummy_batch, dummy_output_dim])

    # ---------------------------
    # Verify that all models will have output
    # shape=(batch, dummy_output_dim)
    # ---------------------------
    model_names = ("Inception", "EEGITNet", "Deep4Net", "EEGNetv1", "EEGNetv4", "EEGResNet", "EEGInception",
                   "SleepStagerChambon2018", "TIDNet")
    for model_name in model_names:
        # Define model. These kwargs should be sufficient (although redundant in some cases)
        model = MTSClassifier(classifier_name=model_name, in_channels=dummy_channels, num_classes=dummy_output_dim,
                              time_steps=dummy_time_steps, sampling_freq=dummy_sampling_freq)

        # Forward pass
        outputs = model(dummy_data)

        # Check dimensions
        output_size = outputs.size()
        assert output_size == expected_output_size, f"The model {model_name} had an expected output with " \
                                                    f"size={expected_output_size}, but returned an output with " \
                                                    f"size={output_size}"


def test_correct_classifiers() -> None:
    """Test that the correct classifier is used for all implemented classifiers"""
    # -----------------------
    # Make dummy hyperparameters
    # -----------------------
    dummy_batch, dummy_channels, dummy_time_steps, dummy_sampling_freq = 10, 32, 2000, 128
    dummy_output_dim = 7

    # -----------------------
    # Check all names
    # -----------------------
    model_names = ("Inception", "EEGITNet", "Deep4Net", "EEGNetv1", "EEGNetv4", "EEGResNet", "EEGInception",
                   "SleepStagerChambon2018", "TIDNet")
    for model_name in model_names:
        # Define model. These kwargs should be sufficient (although redundant in some cases)
        model = MTSClassifier(classifier_name=model_name, in_channels=dummy_channels, num_classes=dummy_output_dim,
                              time_steps=dummy_time_steps, sampling_freq=dummy_sampling_freq)

        # Check that the correct classifier is used
        assert type(model._classifier).__name__ in [model_name, f"{model_name}MTS"], \
            f"Expected model {model_name}, but received {type(model._classifier).__name__}"
