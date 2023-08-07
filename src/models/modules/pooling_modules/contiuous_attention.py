from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.datasets.data_base import channel_names_to_indices
from src.models.modules.pooling_modules.pooling_base import GroupPoolingBase


class ContinuousAttention(GroupPoolingBase):
    """
    Continuous attention using blocks from InceptionTime. Skip connections are not used. No head region is used
    """

    supports_precomputing = False
    forward_args = "x", "ch_names", "channel_name_to_index"

    def __init__(self, num_regions: Optional[int] = None, share_pooling_function: Optional[bool] = None,
                 cnn_units: int = 32, depth: int = 6, use_bottleneck: bool = True,
                 activation: Optional[Callable] = None, max_kernel_size: int = 40):
        if share_pooling_function is None:
            share_pooling_function = True if num_regions is None else False

        super().__init__()

        # -----------------------------
        # Define Inception modules. One per region,
        # or one for all
        # -----------------------------
        if share_pooling_function or num_regions is None:
            self._inception_modules = nn.ModuleList([
                _InceptionModule(cnn_units=cnn_units, depth=depth, use_bottleneck=use_bottleneck, activation=activation,
                                 max_kernel_size=max_kernel_size)])
        else:
            self._inception_modules = nn.ModuleList([
                _InceptionModule(cnn_units=cnn_units, depth=depth, use_bottleneck=use_bottleneck, activation=activation,
                                 max_kernel_size=max_kernel_size) for _ in range(num_regions)])

        self._share_pooling_function = share_pooling_function

    def forward(self, x: torch.Tensor, ch_names: Tuple[Tuple[str, ...], ...], channel_name_to_index: Dict[str, int]) \
            -> torch.Tensor:
        """
        Forward method
        Args:
            x: Full EEG data
            ch_names: Channel names of regions
            channel_name_to_index: Dict mapping channel name to index

        Returns: Region representations after applying continuous attention
        Example:
            >>> my_cs_names = (("Cz", "POO10h", "FFT7h"), ("C3", "C1"), ("PPO10h", "POO10h", "FTT7h", "FTT7h"),
            ...                ("C3", "C1"), ("PPO10h", "POO10h", "Cz"), ("Cz",))
            >>> my_channel_name_to_index = {"Cz": 0, "C1": 1, "C3": 2, "PPO10h": 3, "POO10h": 4, "FTT7h": 5, "FFT7h": 6,
            ...                             "Fp1": 7}
            >>> my_x = torch.rand(size=(10, 13, 300))
            >>> my_model = ContinuousAttention(depth=2)
            >>> my_model(x=my_x, ch_names=my_cs_names, channel_name_to_index=my_channel_name_to_index).size()
            torch.Size([10, 6, 300])
            >>> my_model = ContinuousAttention(depth=2, num_regions=6)
            >>> my_model(x=my_x, ch_names=my_cs_names, channel_name_to_index=my_channel_name_to_index).size()
            torch.Size([10, 6, 300])
            >>> my_model = ContinuousAttention(depth=2, num_regions=7)
            >>> my_model(x=my_x, ch_names=my_cs_names, channel_name_to_index=my_channel_name_to_index).size()
            Traceback (most recent call last):
            ...
            AssertionError: Expected 7 number of regions, but received 6
        """
        if self._share_pooling_function:
            inception_modules = cycle(self._inception_modules)
        else:
            # Number of regions should be as expected
            assert len(ch_names) == len(self._inception_modules), f"Expected {len(self._inception_modules)} number " \
                                                                  f"of regions, but received {len(ch_names)}"

            inception_modules = self._inception_modules

        # Loop through all regions
        region_representations: List[torch.Tensor] = list()
        for legal_ch_names, inception_module in zip(ch_names, inception_modules):
            # Extract indices of the nodes in the group
            allowed_node_indices = channel_names_to_indices(channel_names=legal_ch_names,
                                                            channel_name_to_index=channel_name_to_index)

            # Send through Inception module
            attention = inception_module(x[:, allowed_node_indices])

            # Global average pooling in feature dimension
            attention = torch.mean(attention, dim=1)  # shape=(batch, channels, time_steps)

            # L1 normalise (such that it actually is attention)
            attention = torch.softmax(attention, dim=1)

            # Attention is applied by Hadamard product and average
            region_representations.append(torch.mean(attention*x[:, allowed_node_indices], dim=1, keepdim=True))

        # Concatenate and return
        return torch.cat(region_representations, dim=1)


# ---------------------------
# Sub-modules
# ---------------------------
class _InceptionModule(nn.Module):
    """
    Using blocks from InceptionTime. Skip connections are not used
    """

    def __init__(self, cnn_units: int = 32, depth: int = 6, use_bottleneck: bool = True,
                 activation: Optional[Callable] = None, max_kernel_size: int = 40):
        super().__init__()

        # -----------------------------
        # Define Inception modules
        # -----------------------------
        output_channels = cnn_units * (_InceptionSubModule.num_kernel_sizes + 1)
        self._inception_sub_modules = nn.ModuleList(
            [_InceptionSubModule(in_channels=in_channel, units=cnn_units, use_bottleneck=use_bottleneck,
                                 activation=activation, max_kernel_size=max_kernel_size)
             for i, in_channel in enumerate([1] + [output_channels] * (depth - 1))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: A torch.Tensor with shape=(batch, timeseries per eeg_channel, eeg_channels, time steps)

        Returns:
            Output of inception module
        Examples:
            >>> my_inception_module = _InceptionModule()
            >>> _ = my_inception_module.eval()
            >>> my_data = torch.rand(size=(10, 53, 600))
            >>> my_output = my_inception_module(my_data)
            >>> my_output.size()
            torch.Size([10, 128, 53, 600])
            >>> # Verify that the code runs with different number of channels
            >>> my_inception_module(torch.rand(size=(10, 173, 600))).size()
            torch.Size([10, 128, 173, 600])
            >>> # Verify that a change to one channel does not affect the output of the others
            >>> my_data_2 = my_data
            >>> my_data_2[:, 0] = torch.rand(size=(10, 600))
            >>> my_output_2 = my_inception_module(my_data)
            >>> torch.equal(my_output[:, :, 1:], my_output_2[:, :, 1:])  # The other channels are not affected
            True
            >>> torch.equal(my_output[:, :, 0], my_output_2[:, :, 0])  # The one channel is affected
            False
            >>> # Verify that the code runs with different arguments:
            >>> _InceptionModule(cnn_units=11, activation=F.relu, use_bottleneck=False,
            ...                  max_kernel_size=8)(my_data).size()
            torch.Size([10, 44, 53, 600])
        """
        # Expand dimensions
        if x.dim() == 3:
            x = torch.unsqueeze(x, dim=1)

        # ------------------
        # Pass through all sub-modules
        # ------------------
        for inception_sub_module in self._inception_sub_modules:
            x = inception_sub_module(x)
        return x


class _InceptionSubModule(nn.Module):

    num_kernel_sizes = 3

    def __init__(self, in_channels: int, units: int = 32, bottleneck_units: int = 32,
                 activation: Optional[Callable] = None, use_bottleneck: bool = True, max_kernel_size: int = 40):
        """
        Initialise

        As opposed to the original keras implementation, strides is strictly set to 1 and cannot be specified to any
        other value. This is because setting padding='same' is not supported when strides are greater than 1
        Args:
            in_channels: Number of expected input channels
            units: Output (channel) dimension of the Conv layers. Equivalent to nb_filters in original keras
                implementation
            activation: Activation function. If None is passed, no activation function will be used
            use_bottleneck: To use the first input_conv layer or not
            max_kernel_size: Largest kernel size used. In the original keras implementation, the equivalent argument is
                stored as kernel_size - 1, the same is not done here
        """
        super().__init__()

        # Store selected activation function
        self._activation_function = _no_activation_function if activation is None else activation

        # -------------------------------
        # Define Conv layer maybe operating on
        # the input
        # -------------------------------
        if use_bottleneck:
            self._input_conv = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_units, kernel_size=1,
                                         padding="same", bias=False)
            out_channels = bottleneck_units
        else:
            self._input_conv = None
            out_channels = in_channels

        # -------------------------------
        # Define convolutional layers with different
        # kernel sizes (to be concatenated at the end)
        # -------------------------------
        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(self.num_kernel_sizes)]

        self._conv_list = nn.ModuleList([nn.Conv2d(in_channels=out_channels, out_channels=units,
                                                   kernel_size=(1, kernel_size), stride=1, padding="same", bias=False)
                                         for kernel_size in kernel_sizes])

        # -------------------------------
        # Define Max pooling and conv layer to be
        # applied after max pooling
        # -------------------------------
        self._max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self._conv_after_max_pool = nn.Conv2d(in_channels=in_channels, out_channels=units, kernel_size=(1, 1),
                                              padding="same", bias=False)

        # Finally, define batch norm
        self._batch_norm = nn.BatchNorm2d(num_features=units * (len(self._conv_list) + 1))  # Must multiply due to
        # concatenation with all outputs from self._conv_list and self._con_after_max_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: A torch.Tensor with shape=(batch, timeseries per eeg_channel, eeg_channels, time steps)
        Returns:
            Output of inception submodule
        Examples:
            >>> my_in_chans = 7
            >>> my_inception_module = _InceptionSubModule(in_channels=my_in_chans)
            >>> _ = my_inception_module.eval()
            >>> my_data = torch.rand(size=(10, my_in_chans, 53, 600))
            >>> my_output = my_inception_module(my_data)
            >>> my_output.size()
            torch.Size([10, 128, 53, 600])
            >>> # Verify that the code runs with different number of channels
            >>> my_inception_module(torch.rand(size=(10, my_in_chans, 173, 600))).size()
            torch.Size([10, 128, 173, 600])
            >>> # Verify that a change to one channel does not affect the output of the others
            >>> my_data_2 = my_data
            >>> my_data_2[:, :, 0] = torch.rand(size=(10, my_in_chans, 600))
            >>> my_output_2 = my_inception_module(my_data)
            >>> torch.equal(my_output[:, :, 1:], my_output_2[:, :, 1:])  # The other channels are not affected
            True
            >>> torch.equal(my_output[:, :, 0], my_output_2[:, :, 0])  # The one channel is affected
            False
            >>> # Verify that the code runs with different arguments:
            >>> _InceptionSubModule(in_channels=my_in_chans, units=11, activation=F.relu, use_bottleneck=False,
            ...                     max_kernel_size=8)(my_data).size()
            torch.Size([10, 44, 53, 600])
        """
        # Maybe pass through input conv
        if self._input_conv is not None:
            inception_input = self._activation_function(self._input_conv(x))
        else:
            inception_input = torch.clone(x)

        # Pass through the conv layers with different kernel sizes
        outputs = []
        for i, conv_layer in enumerate(self._conv_list):
            outputs.append(self._activation_function(conv_layer(inception_input)))

        # Pass input tensor through max pooling, followed by a conv layer
        max_pool_output = self._max_pool(x)
        outputs.append(self._activation_function(self._conv_after_max_pool(max_pool_output)))

        # Concatenate, add batch norm, apply Relu activation function and return
        x = torch.cat(outputs, dim=1)  # concatenate in channel dimension
        x = F.relu(self._batch_norm(x))

        return x


# ------------
# Functions
# ------------
def _no_activation_function(x: torch.Tensor) -> torch.Tensor:
    """This can be used as activation function if no activation function is wanted. It is typically more convenient to
    use this function, instead of handling activation functions of type None"""
    return x


def _pad_and_stack_2d(regions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Pad and stack regions into a single torch tensor
    Args:
        regions: EEG data of all regions

    Returns: A torch tensor with shape=(batch, num_regions, max(num_channels), time_steps)
    Examples:
        >>> batch, time_steps = 10, 300
        >>> my_regions = (torch.rand(size=(batch, 5, time_steps)), torch.rand(size=(batch, 3, time_steps)),
        ...               torch.rand(size=(batch, 2, time_steps)), torch.rand(size=(batch, 7, time_steps)),
        ...               torch.rand(size=(batch, 3, time_steps)))
        >>> _pad_and_stack_2d(regions=my_regions).size()  # 5 regions, the highest number of channels is 7
        torch.Size([10, 5, 7, 300])
        >>> batch, time_steps = 2, 10
        >>> my_regions = (torch.ones(size=(batch, 3, time_steps)), torch.ones(size=(batch, 1, time_steps))*2,
        ...               torch.ones(size=(batch, 2, time_steps))*3, torch.ones(size=(batch, 4, time_steps))*4,
        ...               torch.ones(size=(batch, 3, time_steps))*5)
        >>> _pad_and_stack_2d(regions=my_regions)[0]
        tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                 [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                 [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                 [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
                 [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]],
        <BLANKLINE>
                [[5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                 [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                 [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    """
    # Get the number of channels in all regions
    num_channels = _get_num_channels(regions)
    max_num_channels = max(num_channels)

    padded_regions = [torch.unsqueeze(F.pad(region, (0, 0, 0, max_num_channels - num_chans), mode="constant",
                                            value=0.0), dim=1) for region, num_chans in zip(regions, num_channels)]

    # Concatenate and return
    return torch.cat(padded_regions, dim=1)


def _get_num_channels(regions: Tuple[torch.Tensor, ...]) -> Tuple[int, ...]:
    """
    Get the number of channels in all regions
    Args:
        regions: EEG data from regions

    Returns: A tuple of number of channels
    Examples:
        >>> batch, time_steps = 10, 300
        >>> my_regions = (torch.rand(size=(batch, 5, time_steps)), torch.rand(size=(batch, 3, time_steps)),
        ...               torch.rand(size=(batch, 2, time_steps)), torch.rand(size=(batch, 7, time_steps)),
        ...               torch.rand(size=(batch, 3, time_steps)))
        >>> _get_num_channels(regions=my_regions)
        (5, 3, 2, 7, 3)
    """
    return tuple([region.size()[1] for region in regions])
