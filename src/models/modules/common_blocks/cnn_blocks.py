"""
Implementing commonly used CNN blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNNBlock1(nn.Module):
    """
    Two 1d CNN layers with relu as activation function, followed by max pooling in the temporal dimension (1x2)
    """
    __slots__ = "_conv1", "_conv2", "_max_pool"

    def __init__(self, **kwargs):
        """Kwargs must contain the following: in_channels, units, kernel_width. Other arguments can also be used"""
        super().__init__()

        # --------------------------------------
        # Required args:
        # --------------------------------------
        in_channels = kwargs.pop("in_channels")
        units = kwargs.pop("units")

        kernel_width = kwargs.pop("kernel_width")

        # --------------------------------------
        # Optional args (run with default if
        # not specified):
        # --------------------------------------
        groups = kwargs.get("groups")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")
        padding_mode = kwargs.get("padding_mode")
        dilation = kwargs.get("dilation")

        # Set to torch.nn default (as per 8th of August 2022)
        groups = 1 if groups is None else groups
        stride = 1 if stride is None else stride
        padding = 0 if padding is None else padding
        padding_mode = "zeros" if padding_mode is None else padding_mode
        dilation = 1 if dilation is None else dilation

        # --------------------------------------
        # Define two Conv-layers and a max-pooling
        # --------------------------------------
        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=units, kernel_size=kernel_width, groups=groups,
                                stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation)
        self._conv2 = nn.Conv1d(in_channels=units, out_channels=units, kernel_size=kernel_width, groups=groups,
                                stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation)

        self._max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param x: torch.Tensor with shape=(batch, channels, time_steps)
        :return: Output of CNN block
        Examples:
            >>> my_data = torch.rand(size=(10, 13, 300))
            >>> my_model = _CNNBlock1(in_channels=13, units=43, kernel_width=3)
            >>> my_model(my_data).size()
            torch.Size([10, 43, 148])
            >>> my_data = torch.rand(size=(10, 14, 1500))
            >>> my_model = _CNNBlock1(in_channels=14, units=49, kernel_width=3, stride=2, groups=7, dilation=1)
            >>> my_model(my_data).size()
            torch.Size([10, 49, 187])
        """
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))

        x = self._max_pool(x)
        return x


class _CNNBlock2(nn.Module):
    """
    Two 1d CNN layers with relu as activation function, followed by average pooling in the temporal dimension (1x2)
    """
    __slots__ = "_conv1", "_conv2", "_avg_pool"

    def __init__(self, **kwargs):
        """Kwargs must contain the following: in_channels, units, kernel_width. Other arguments can also be used"""
        super().__init__()

        # --------------------------------------
        # Required args:
        # --------------------------------------
        in_channels = kwargs.pop("in_channels")
        units = kwargs.pop("units")

        kernel_width = kwargs.pop("kernel_width")

        # --------------------------------------
        # Optional args (run with default if
        # not specified):
        # --------------------------------------
        groups = kwargs.get("groups")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")
        padding_mode = kwargs.get("padding_mode")
        dilation = kwargs.get("dilation")

        # Set to torch.nn default (as per 8th of August 2022)
        groups = 1 if groups is None else groups
        stride = 1 if stride is None else stride
        padding = 0 if padding is None else padding
        padding_mode = "zeros" if padding_mode is None else padding_mode
        dilation = 1 if dilation is None else dilation

        # --------------------------------------
        # Define two Conv-layers and a max-pooling
        # --------------------------------------
        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=units, kernel_size=kernel_width, groups=groups,
                                stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation)
        self._conv2 = nn.Conv1d(in_channels=units, out_channels=units, kernel_size=kernel_width, groups=groups,
                                stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation)

        self._avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param x: torch.Tensor with shape=(batch, channels, time_steps)
        :return: Output of CNN block
        Examples:
            >>> my_data = torch.rand(size=(10, 13, 300))
            >>> my_model = _CNNBlock2(in_channels=13, units=43, kernel_width=3)
            >>> my_model(my_data).size()
            torch.Size([10, 43, 148])
            >>> my_data = torch.rand(size=(10, 14, 1500))
            >>> my_model = _CNNBlock2(in_channels=14, units=49, kernel_width=3, stride=2, groups=7, dilation=1)
            >>> my_model(my_data).size()
            torch.Size([10, 49, 187])
        """
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))

        x = self._avg_pool(x)
        return x


class _CNNBlock3(nn.Module):
    """
    ResNet block. Two 1D CNN layers with relu as activation function, with skip connection
    """

    def __init__(self, **kwargs) -> None:
        """Kwargs must contain the following: in_channels, kernel_width. Other arguments can also be used"""
        super().__init__()

        # --------------------------------------
        # Required args
        # --------------------------------------
        in_channels: int = kwargs.pop("in_channels")
        kernel_width: int = kwargs.pop("kernel_width")

        # --------------------------------------
        # Optional args (run with PyTorch default if
        # not specified):
        # --------------------------------------
        groups: int = kwargs.get("groups", 1)
        stride: int = kwargs.get("stride", 1)
        dilation = kwargs.get("dilation", 1)

        # --------------------------------------
        # Define two conv layers
        # --------------------------------------
        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_width,
                                groups=groups, stride=stride, padding="same", dilation=dilation)
        self._conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_width,
                                groups=groups, stride=stride, padding="same", dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param x: torch.Tensor with shape=(batch, channels, time_steps)
        :return: Output of CNN block, which has the same shape as the input
        Examples:
            >>> my_model = _CNNBlock3(in_channels=77, kernel_width=5, dilation=3)
            >>> my_input = torch.rand(size=(10, 77, 300))
            >>> my_model(my_input).size()
            torch.Size([10, 77, 300])
        """
        # Store the input
        input_x = x

        # Pass through the two conv layers with relu as activation
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))

        # return the sum
        return x + input_x


class CNNBlock(nn.Module):

    __slots__ = "_cnn_block"

    def __init__(self, block_id: int, **kwargs):
        """
        initialise
        :param block_id: Which of the implemented CNN-blocks to use
        :param kwargs: Arguments which should be passed to the CNN-block, such as input channels and output channels
        Examples:
            >>> my_model = CNNBlock(block_id=1, in_channels=13, units=43, kernel_width=7)
            >>> type(my_model._cnn_block)
            <class 'cnn_blocks._CNNBlock1'>
            >>> my_model = CNNBlock(block_id=2, in_channels=13, units=43, kernel_width=7, stride=3)
            >>> type(my_model._cnn_block)
            <class 'cnn_blocks._CNNBlock2'>
            >>> my_model = CNNBlock(block_id=3, in_channels=13, kernel_width=7)
            >>> type(my_model._cnn_block)
            <class 'cnn_blocks._CNNBlock3'>
            >>> _ = CNNBlock(block_id=-1, in_channels=13, units=43, kernel_width=7)
            Traceback (most recent call last):
            ...
            ValueError: The block id -1 was not recognised
        """
        super().__init__()

        self._cnn_block = self._get_cnn_block(block_id=block_id, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(Tested separately for each class)"""
        return self._cnn_block(x)

    @staticmethod
    def _get_cnn_block(block_id: int, **kwargs) -> nn.Module:
        if block_id == 1:
            return _CNNBlock1(**kwargs)
        elif block_id == 2:
            return _CNNBlock2(**kwargs)
        elif block_id == 3:
            return _CNNBlock3(**kwargs)
        else:
            raise ValueError(f"The block id {block_id} was not recognised")
