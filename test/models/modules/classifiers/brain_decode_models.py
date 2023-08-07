"""
https://braindecode.org/stable/api.html#models
"""
import torch
from braindecode import models


def test_output_dimensions() -> None:
    """
    Test that forward method of all models in braindecode can accept torch.Tensor with size=(batch, channels,
    time_steps), and that the output has expected dimensions. Note that some models classify over time (output also
    contain a temporal dimension)

    Note that the following do not pass the test: HybridNet, ShallowFBCSPNet, TCN, SleepStagerEldele2021
    """
    # Define input data
    dummy_sfreq = 128
    dummy_batch, dummy_channels, dummy_time_steps = 10, 65, dummy_sfreq * 30
    dummy_data = torch.rand(size=(dummy_batch, dummy_channels, dummy_time_steps))

    # Define shared hyperparameters
    output_dim = 3
    expected_output_size = torch.Size([dummy_batch, output_dim])

    # ---------------------------------
    # Test all braindecode models
    # ---------------------------------
    model = models.EEGITNet(n_classes=output_dim, in_channels=dummy_channels, input_window_samples=dummy_time_steps)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'EEGITNet'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.Deep4Net(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps,
                            final_conv_length="auto")
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'Deep4Net'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.EEGNetv4(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'EEGNetv4'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.EEGNetv1(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'EEGNetv1'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.HybridNet(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps)
    output_size = model(dummy_data).size()  # looks like one output for each time-step (with some edge cutting due)
    assert output_size == torch.Size([expected_output_size[0], expected_output_size[1], 3319]), \
        f"Model: {'HybridNet'}\n" \
        f"Expected: {torch.Size([expected_output_size[0], expected_output_size[1], 3319])}\n" \
        f"Received: {output_size}"

    model = models.ShallowFBCSPNet(in_chans=dummy_channels, n_classes=output_dim)
    output_size = model(dummy_data).size()  # This one is strange, but probably at different time points
    assert output_size[:2] == expected_output_size, f"Model: {'ShallowFBCSPNet'}\nExpected: {expected_output_size}\n" \
                                                    f"Received: {output_size[:2]}"

    model = models.EEGResNet(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps,
                             final_pool_length="auto", n_first_filters=dummy_channels * 2)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'EEGResNet'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.EEGInception(in_channels=dummy_channels, n_classes=output_dim,
                                input_window_samples=dummy_time_steps)  # num time steps is stupidly enough needed here
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'EEGInception'}\nExpected: {expected_output_size}\n" \
                                                f"Received: {output_size}"

    model = models.TCN(n_in_chans=dummy_channels, n_outputs=output_dim, n_blocks=3, n_filters=dummy_channels * 2,
                       kernel_size=5, add_log_softmax=True, drop_prob=0.5)
    output_size = model(dummy_data).size()  # looks like one output for each time-step (with some edge cutting due)
    assert output_size == torch.Size([expected_output_size[0], expected_output_size[1], 3784]), \
        f"Model: {'TCN'}\nExpected: {torch.Size([expected_output_size[0], expected_output_size[1], 3784])}\n" \
        f"Received: {output_size}"

    model = models.SleepStagerChambon2018(n_channels=dummy_channels, sfreq=dummy_sfreq, input_size_s=30,
                                          n_classes=output_dim)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'SleepStagerChambon2018'}\n" \
                                                f"Expected: {expected_output_size}\nReceived: {output_size}"

    # This one will not be relevant to use
    model = models.SleepStagerBlanco2020(n_channels=dummy_channels, sfreq=dummy_sfreq, input_size_s=30,
                                         n_groups=dummy_channels, n_conv_chans=dummy_channels*5, n_classes=output_dim)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'SleepStagerBlanco2020'}\n" \
                                                f"Expected: {expected_output_size}\nReceived: {output_size}"

    # Damn, these models have strange properties. Single channeled and only 100 Hz at 30 seconds...
    model = models.SleepStagerEldele2021(sfreq=100, n_classes=output_dim)
    output_size = model(dummy_data[..., :1, :100*30]).size()
    assert output_size == expected_output_size, f"Model: {'SleepStagerEldele2021'}\n" \
                                                f"Expected: {expected_output_size}\nReceived: {output_size}"

    model = models.TIDNet(in_chans=dummy_channels, n_classes=output_dim, input_window_samples=dummy_time_steps)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'TIDNet'}\n" \
                                                f"Expected: {expected_output_size}\nReceived: {output_size}"

    model = models.USleep(in_chans=dummy_channels, sfreq=dummy_sfreq, n_classes=output_dim)
    output_size = model(dummy_data).size()
    assert output_size == expected_output_size, f"Model: {'USleep'}\n" \
                                                f"Expected: {expected_output_size}\nReceived: {output_size}"
