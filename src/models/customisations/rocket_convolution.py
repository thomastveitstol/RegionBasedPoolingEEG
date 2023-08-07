"""
Classes for using ROCKET convolutions

TODO: (1) stochasticity in __init__ method is not good for loading the modules. (2) try not to append to lists, if
    possible. (3) make a summary of when to use which implementation
"""
import random
from typing import List, Tuple, Union

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class RocketConv1d(nn.Module):
    # TODO: the stochasticity in the __init__ method gives mismatch in saving and loading
    def __init__(self,
                 num_kernels: int,
                 max_receptive_field: int):
        super().__init__()

        # Initialise list of kernels
        kernels = []

        # ------------------
        # Define kernels
        # ------------------
        for _ in range(num_kernels):
            # Sample dilation and kernel length
            kernel_length = _sample_kernel_length()
            dilation = _sample_dilation(max_receptive_field=max_receptive_field, kernel_length=kernel_length)

            # Define kernel
            rocket_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_length, dilation=dilation,
                                      padding="same", groups=1)

            # Initialise weights
            _sample_weights(weights=rocket_kernel.weight.data)
            _sample_bias(bias=rocket_kernel.bias.data)

            # Add to kernels list
            kernels.append(rocket_kernel)

        # Register kernels using module list
        self._kernels = nn.ModuleList(kernels)

        # ------------------
        # Freeze all parameters
        # ------------------
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: A torch tensor with shape=(batch, 1, channels, time_steps)

        Returns: Features
        Examples:
            >>> my_model = RocketConv1d(num_kernels=321, max_receptive_field=123)
            >>> my_model(torch.rand(size=(10, 1, 64, 500))).size()
            torch.Size([10, 64, 642])
        """
        if x.dim() == 4:
            x = torch.squeeze(x, dim=1)

        # Initialise tensor. The features will be stored to this tensor
        batch, num_channels, _ = x.size()
        outputs = torch.zeros(size=(batch, num_channels, 2*self.num_kernels)).to(x.device)

        # Loop through all kernels
        for i, kernel in enumerate(self._kernels):
            convolution = F.conv1d(input=x,
                                   weight=kernel.weight.data.repeat(num_channels, 1, 1),
                                   bias=kernel.bias.data.repeat(num_channels), stride=1, padding="same",
                                   dilation=kernel.dilation,
                                   groups=num_channels)
            outputs[..., (2*i):(2*i+2)] = compute_ppv_and_max(torch.unsqueeze(convolution, dim=1))

        # Concatenate and return
        return outputs

    # --------------
    # Properties
    # --------------
    @property
    def num_kernels(self) -> int:
        return len(self._kernels)


class RocketConv(nn.Module):

    def __init__(self,
                 num_kernels: int,
                 max_receptive_field: int):
        """
        Initialise
        Args:
            num_kernels: Number of random convolutional kernels
            max_receptive_field: Max receptive field
        Examples:
            >>> random.seed(1)
            >>> numpy.random.seed(2)
            >>> _ = torch.manual_seed(3)
            >>> my_conv = RocketConv(num_kernels=300, max_receptive_field=129)
            >>> my_conv._kernels[0].weight.data.size()
            torch.Size([1, 1, 1, 7])
            >>> my_conv._kernels[1].weight.data.size()
            torch.Size([1, 1, 1, 11])
            >>> my_conv._kernels[3].weight.data.size()
            torch.Size([1, 1, 1, 9])
            >>> all(not my_param.requires_grad for my_param in my_conv.parameters())  # type: ignore[attr-defined]
            True
        """
        super().__init__()

        # Initialise list of kernels
        kernels = []

        # ------------------
        # Define kernels
        # ------------------
        for _ in range(num_kernels):
            # Sample dilation and kernel length
            kernel_length = _sample_kernel_length()
            dilation = _sample_dilation(max_receptive_field=max_receptive_field, kernel_length=kernel_length)

            # Define kernel
            rocket_kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_length), dilation=dilation,
                                      padding="same")

            # Initialise weights
            _sample_weights(weights=rocket_kernel.weight.data)
            _sample_bias(bias=rocket_kernel.bias.data)

            # Add to kernels list
            kernels.append(rocket_kernel)

        # Register kernels using module list
        self._kernels = nn.ModuleList(kernels)

        # ------------------
        # Freeze all parameters
        # ------------------
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: A torch tensor with shape=(batch, 1, channels, time_steps)

        Returns: Features
        Examples:
            >>> my_model = RocketConv(num_kernels=321, max_receptive_field=123)
            >>> my_model(torch.rand(size=(10, 1, 64, 500))).size()
            torch.Size([10, 64, 642])
        """
        # Initialise list. The features will be appended to this list
        outputs: List[torch.Tensor] = []

        # Loop through all kernels
        for kernel in self._kernels:
            outputs.append(compute_ppv_and_max(kernel(x)))

        # Concatenate and return
        return torch.cat(outputs, dim=-1).to()


class RocketConv2d(nn.Conv2d):
    """
    Class for convolutions as performed in ROCKET.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 padding: Union[str, int, Tuple[int, int]] = "same",
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device=None,
                 dtype=None
                 ):
        """
        Initialise. See parent class for documentation on input arguments
        Examples:
            >>> random.seed(1)
            >>> numpy.random.seed(2)
            >>> _ = torch.manual_seed(3)
            >>> my_conv = RocketConv2d(in_channels=1, out_channels=300, kernel_size=(1, 50))
            >>> my_conv.weight.data.size()
            torch.Size([300, 1, 1, 50])
            >>> torch.nonzero(my_conv.weight.data[7, 0, 0], as_tuple=True)[0]
            tensor([13, 16, 19, 22, 25, 28, 31, 34, 37])
            >>> torch.nonzero(my_conv.weight.data[34, 0, 0], as_tuple=True)[0]
            tensor([19, 21, 23, 25, 27, 29, 31])
            >>> all(not my_param.requires_grad for my_param in my_conv.parameters())  # type: ignore[attr-defined]
            True
        """
        # ------------------
        # Initialise
        # ------------------
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode,
                         device=device, dtype=dtype)

        # ------------------
        # Freeze all parameters
        # ------------------
        for param in self.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        """
        Initialise parameters
        Examples:
            >>> random.seed(1)
            >>> numpy.random.seed(2)
            >>> _ = torch.manual_seed(3)
            >>> my_conv = RocketConv2d(in_channels=1, out_channels=300, kernel_size=(1, 50))
            >>> my_conv.weight.data.size()
            torch.Size([300, 1, 1, 50])
            >>> my_conv.weight.data[0, 0, 0]
            tensor([-0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,
                    -0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                     0.0000,  0.0000, -0.0000, -1.6562,  0.0000,  0.5176, -0.0000,  0.0485,
                     0.0000,  2.2520,  0.0000, -1.1998,  0.0000,  0.4546,  0.0000, -0.1513,
                    -0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,
                    -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000,
                    -0.0000,  0.0000])
            >>> torch.nonzero(my_conv.weight.data[7, 0, 0], as_tuple=True)[0]
            tensor([13, 16, 19, 22, 25, 28, 31, 34, 37])
            >>> torch.nonzero(my_conv.weight.data[34, 0, 0], as_tuple=True)[0]
            tensor([19, 21, 23, 25, 27, 29, 31])
            >>> torch.nonzero(my_conv.weight.data[67, 0, 0], as_tuple=True)[0]
            tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
            >>> torch.nonzero(my_conv.weight.data[195, 0, 0], as_tuple=True)[0]
            tensor([17, 19, 21, 23, 25, 27, 29, 31, 33])
            >>> torch.nonzero(my_conv.weight.data[299, 0, 0], as_tuple=True)[0]
            tensor([ 7, 13, 19, 25, 31, 37, 43])
        """
        # ---------------------------
        # Initialise weights and bias
        # ---------------------------
        _sample_weights(weights=self.weight.data)
        _sample_bias(bias=self.bias.data)

        # ---------------------------
        # Sample kernel lengths and dilations
        # ---------------------------
        num_kernels = self.weight.size()[0]
        max_receptive_field = self.weight.size()[-1]

        kernel_lengths = tuple(_sample_kernel_length() for _ in range(num_kernels))
        dilations = tuple(_sample_dilation(max_receptive_field=max_receptive_field,
                                           kernel_length=kernel_length) for kernel_length in kernel_lengths)

        # ---------------------------
        # Zero-fill weights and set to sparse
        # ---------------------------
        _zero_fill(weights=self.weight.data, kernel_lengths=kernel_lengths, dilations=dilations)


# ---------------------
# Functions for zero filling
# ---------------------
def _zero_fill_single_tensor(weights: torch.Tensor, kernel_length: int, dilation: int):
    """
    Function for zero filling tensor with shape=(1, max_receptive_field). The change of the parameters also happen
    in-place
    Examples:
        >>> _ = torch.manual_seed(2)
        >>> my_weights = torch.rand(17)
        >>> _zero_fill_single_tensor(weights=my_weights, kernel_length=7, dilation=2)
        tensor([0.0000, 0.0000, 0.6371, 0.0000, 0.7136, 0.0000, 0.4425, 0.0000, 0.6142,
                0.0000, 0.5657, 0.0000, 0.3901, 0.0000, 0.5334, 0.0000, 0.0000])
        >>> my_weights
        tensor([0.0000, 0.0000, 0.6371, 0.0000, 0.7136, 0.0000, 0.4425, 0.0000, 0.6142,
                0.0000, 0.5657, 0.0000, 0.3901, 0.0000, 0.5334, 0.0000, 0.0000])
        >>> _zero_fill_single_tensor(weights=torch.rand(14), kernel_length=7, dilation=2)
        tensor([0.0000, 0.3078, 0.0000, 0.0103, 0.0000, 0.4604, 0.0000, 0.4525, 0.0000,
                0.4760, 0.0000, 0.2166, 0.0000, 0.0458])
        >>> _zero_fill_single_tensor(weights=torch.rand(9), kernel_length=3, dilation=3)
        tensor([0.0000, 0.6177, 0.0000, 0.0000, 0.2708, 0.0000, 0.0000, 0.3892, 0.0000])
        >>> torch.nonzero(_zero_fill_single_tensor(weights=torch.rand(18), kernel_length=3, dilation=3),
        ...               as_tuple=True)[0]
        tensor([ 6,  9, 12])
    """
    # Find the index of the element in the middle. The non-zero weighs be be placed with this index as centre
    middle_index = weights.size()[-1] // 2

    # Assuming that kernels length must be odd, as in the original paper
    non_zero_indices = range(middle_index - kernel_length//2 * dilation, middle_index + kernel_length//2 * dilation + 1,
                             dilation)
    zero_indices = [i for i in range(weights.size()[-1]) if i not in non_zero_indices]

    # Set weights to zeros
    weights[zero_indices] *= 0

    return weights


def _zero_fill(weights: torch.Tensor, kernel_lengths: Tuple[int, ...], dilations: Tuple[int, ...]):
    """
    Zero-fill weights. The change is also made in-place
    Examples:
        >>> _ = torch.manual_seed(2)
        >>> my_weights = torch.rand(size=(4, 1, 1, 44))
        >>> my_kernel_lengths = 7, 9, 7, 11
        >>> my_dilations = 5, 2, 6, 4
        >>> my_zero_filled = _zero_fill(weights=my_weights, kernel_lengths=my_kernel_lengths, dilations=my_dilations)
        >>> torch.nonzero(my_zero_filled[0, 0, 0], as_tuple=True)[0]
        tensor([ 7, 12, 17, 22, 27, 32, 37])
        >>> torch.nonzero(my_zero_filled[1, 0, 0], as_tuple=True)[0]
        tensor([14, 16, 18, 20, 22, 24, 26, 28, 30])
        >>> torch.nonzero(my_zero_filled[2, 0, 0], as_tuple=True)[0]
        tensor([ 4, 10, 16, 22, 28, 34, 40])
        >>> torch.nonzero(my_zero_filled[3, 0, 0], as_tuple=True)[0]
        tensor([ 2,  6, 10, 14, 18, 22, 26, 30, 34, 38, 42])
        >>> torch.equal(my_zero_filled, my_weights)  # In-place
        True
    """
    # Input checks:
    # Verify that number of kernel lengths and dilations is equal to number of kernels
    if not (weights.size()[0] == len(kernel_lengths) == len(dilations)):
        raise ValueError(f"Expected number of kernels to be equal to number of kernel lengths and dilations, but found "
                         f"{weights.size()[1]}, {len(kernel_lengths), len(dilations)}")

    # Dimension check
    if not weights.size()[1] == 1 or not weights.size()[2] == 1:
        raise ValueError(f"Expected first dimension of weight tensor to be 1, but found {weights.size()[0]}")

    # --------------------
    # Zero-fill the weights
    # --------------------
    for weight_tensor, kernel_length, dilation in zip(weights, kernel_lengths, dilations):
        _zero_fill_single_tensor(weights=weight_tensor[0, 0], kernel_length=kernel_length, dilation=dilation)

    return weights


# ---------------------
# Functions for sampling parameters, including
# hyperparameters
# ---------------------
def _sample_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Sample weights such as in the paper. The changes to the tensor is both in-place and returned
    The sampling of weights are done in two steps:
        1) sample every weight from a normal distribution, w ~ N(0, 1)
        2) Mean centre the weights, w = W - mean(W)
    Args:
        weights: Weights to be initialised. In the future: consider passing only the shape
    Returns: An initialised tensor
    Examples:
        >>> _ = torch.random.manual_seed(4)
        >>> my_weights = torch.empty(3, 5)
        >>> _sample_weights(weights=my_weights)
        tensor([[-1.9582, -0.1204,  1.8870,  0.4944,  0.8478],
                [-0.7544, -1.7789,  0.5511,  0.5028,  0.3360],
                [ 0.5321,  1.4178, -0.4338, -0.3016, -1.2216]])
        >>> my_weights  # The input weights are changed in-place due to mutability
        tensor([[-1.9582, -0.1204,  1.8870,  0.4944,  0.8478],
                [-0.7544, -1.7789,  0.5511,  0.5028,  0.3360],
                [ 0.5321,  1.4178, -0.4338, -0.3016, -1.2216]])
        >>> bool(torch.isclose(torch.mean(my_weights), torch.tensor(0.0, dtype=torch.float), atol=1e-7))
        True
    """
    # Step 1) Sample weights from a normal distribution N(0, 1)
    weights = nn.init.normal_(weights, mean=0, std=1)

    # Step 2) Mean centre weights
    weights -= torch.mean(weights)

    return weights


def _sample_bias(bias: torch.Tensor) -> torch.Tensor:
    """
    Sample bias parameters. As in the paper, the weights are sampled from a uniform distribution, b ~ U(-1, 1). Note
    that the initialisation also happens in-place
    Args:
        bias: a bias tensor. Consider only using passing in input shape
    Returns: Bias parameters
    Examples:
        >>> _ = torch.random.manual_seed(4)
        >>> my_bias =  torch.empty(9)
        >>> _sample_bias(my_bias)
        tensor([ 0.1193,  0.1182, -0.8171, -0.5800, -0.9856, -0.9221,  0.9858,  0.8262,
                 0.2372])
        >>> my_bias  # In-place initialisation as well
        tensor([ 0.1193,  0.1182, -0.8171, -0.5800, -0.9856, -0.9221,  0.9858,  0.8262,
                 0.2372])
    """
    return nn.init.uniform_(bias, a=-1, b=1)


def _sample_kernel_length() -> int:
    """
    Following the original paper, the kernel length is selected randomly from {7, 9, 11} with equal probability. Note
    that by 'length' in this context, the number of elements is meant. That is, not taking dilation into account
    Returns: A value in {7, 9, 11}
    Examples:
        >>> random.seed(6)
        >>> _sample_kernel_length()
        11
        >>> _sample_kernel_length()
        7
        >>> _sample_kernel_length()
        9
    """
    return random.choice((7, 9, 11))


def _sample_dilation(max_receptive_field: int, kernel_length) -> int:
    """
    Sample dilation. That is, d = floor(2**x) with x ~ U(0, A) with A as calculated in the paper
    Due to the possibly very long input time series length, it rather uses a max_receptive_filed as upper bound
    Args:
        max_receptive_field: Max receptive field
        kernel_length: length of kernel (in {7, 9, 11} in the paper)
    Returns: Dilation
    Examples:
        >>> numpy.random.seed(3)
        >>> _sample_dilation(max_receptive_field=500, kernel_length=7)
        11
        >>> _sample_dilation(max_receptive_field=1000, kernel_length=9)
        30
    """
    # Set upper bound as in the ROCKET paper, with max_receptive_field instead of input length
    upper_bound = numpy.log2((max_receptive_field - 1) / (kernel_length - 1))

    # Sample from U(0, high)
    x = numpy.random.uniform(low=0, high=upper_bound)

    # Return floor of 2^x
    return int(2**x)


def compute_ppv_and_max(x: torch.Tensor) -> torch.Tensor:
    """
    Compute ppv and max
    Args:
        x: shape=(batch, 1, channels, time_steps)
    Returns: Features of the time series. Output will have shape=(batch, channels, 2) with
        num_features=2 for the current implementation.
    Examples:
        >>> my_data = torch.rand(size=(10, 1, 5, 300))
        >>> compute_ppv_and_max(my_data).size()
        torch.Size([10, 5, 2])
    """
    ppv = torch.mean(torch.heaviside(x, values=torch.tensor(0., dtype=torch.float)), dim=-1)
    max_ = torch.max(x, dim=-1)[0]  # Keep only the values, not the indices

    features = torch.concat([torch.unsqueeze(ppv, dim=-1), torch.unsqueeze(max_, dim=-1)], dim=-1)
    features = torch.permute(features, [0, 2, 1, 3])
    features = features.reshape(features.size()[0], features.size()[1], features.size()[2] * features.size()[3])

    return features
