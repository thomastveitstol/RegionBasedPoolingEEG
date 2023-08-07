"""
Implementing models from BrainDecode. For compatibility purposes, such as setting hyperparameters with kwargs, and
inconsistency with argument names such as in_chans/in_channels, new ones are made.

https://braindecode.org/stable/api.html#models
"""
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import elu
from braindecode.models import EEGITNet, Deep4Net, EEGNetv1, EEGNetv4, EEGResNet, EEGInception, \
    SleepStagerChambon2018, TIDNet
from braindecode.models.functions import identity

from src.models.modules.classifiers.base_classifiers import BaseMTSClassifier


# -----------------------------
# BrainDecode models
# -----------------------------
class EEGITNetMTS(EEGITNet, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")
        time_steps: int = kwargs.pop("time_steps")

        # -----------------------
        # Optional kwargs
        # -----------------------
        dropout_rate: float = kwargs.get("dropout_rate", 0.4)

        # Initialise by calling super class
        super().__init__(n_classes=num_classes, in_channels=in_channels, input_window_samples=time_steps,
                         drop_prob=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(super().forward(x))


class Deep4NetMTS(Deep4Net, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time_steps or final_conv_length must be set
        time_steps: int = kwargs.get("time_steps")
        final_conv_length: Union[int, str] = kwargs.get("final_conv_length", "auto")

        n_filters_time: int = kwargs.get("n_filters_time", 25)
        n_filters_spat: int = kwargs.get("n_filters_spat", 25)
        filter_time_length: int = kwargs.get("filter_time_length", 10)
        pool_time_length: int = kwargs.get("pool_time_length", 3)
        pool_time_stride: int = kwargs.get("pool_time_stride", 3)
        n_filters_2: int = kwargs.get("n_filters_2", 50)
        filter_length_2: int = kwargs.get("filter_length_2", 10)
        n_filters_3: int = kwargs.get("n_filters_3", 100)
        filter_length_3: int = kwargs.get("filter_length_3", 10)
        n_filters_4: int = kwargs.get("n_filters_4", 200)
        filter_length_4: int = kwargs.get("filter_length_4", 10)
        first_conv_nonlin: Callable = kwargs.get("first_conv_nonlin", elu)
        first_pool_mode: str = kwargs.get("first_pool_mode", "max")
        first_pool_nonlin: Callable = kwargs.get("first_pool_nonlin", identity)
        later_conv_nonlin: Callable = kwargs.get("later_conv_nonlin", elu)
        later_pool_mode: str = kwargs.get("later_pool_mode", "max")  # "max" or "mean"
        later_pool_nonlin: Callable = kwargs.get("later_pool_nonlin", identity)
        dropout_rate: float = kwargs.get("dropout_rate", 0.5)
        split_first_layer: bool = kwargs.get("split_first_layer", True)
        batch_norm: bool = kwargs.get("batch_norm", True)
        batch_norm_alpha: float = kwargs.get("batch_norm_alpha", 0.1)
        stride_before_pool: bool = kwargs.get("stride_before_pool", False)

        # Initialise by calling super class
        super().__init__(in_chans=in_channels, n_classes=num_classes, drop_prob=dropout_rate,
                         input_window_samples=time_steps, final_conv_length=final_conv_length,
                         n_filters_time=n_filters_time, n_filters_spat=n_filters_spat,
                         filter_time_length=filter_time_length, pool_time_length=pool_time_length,
                         pool_time_stride=pool_time_stride, n_filters_2=n_filters_2, filter_length_2=filter_length_2,
                         n_filters_3=n_filters_3, filter_length_3=filter_length_3, n_filters_4=n_filters_4,
                         filter_length_4=filter_length_4, first_conv_nonlin=first_conv_nonlin,
                         first_pool_mode=first_pool_mode, first_pool_nonlin=first_pool_nonlin,
                         later_conv_nonlin=later_conv_nonlin, later_pool_mode=later_pool_mode,
                         later_pool_nonlin=later_pool_nonlin, split_first_layer=split_first_layer,
                         batch_norm=batch_norm, batch_norm_alpha=batch_norm_alpha,
                         stride_before_pool=stride_before_pool)


class EEGNetv1MTS(EEGNetv1, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        """
        Initialise
        Args:
            **kwargs: See below
        Examples:
            >>> my_model = EEGNetv1MTS(in_channels=43, num_classes=7, time_steps=2000)
            >>> # Raises AssertionError if both time_steps and final_conv_length are unspecified
            >>> _ = EEGNetv1MTS(in_channels=43, num_classes=7)
            Traceback (most recent call last):
            ...
            AssertionError
        """
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time steps or final conv length must be specified
        time_steps: int = kwargs.get("time_steps")
        final_conv_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        dropout_rate: float = kwargs.get("dropout_rate", 0.25)
        pool_mode: str = kwargs.get("pool_mode", "max")
        second_kernel_size: Tuple[int, int] = kwargs.get("second_kernel_size", (2, 32))
        third_kernel_size: Tuple[int, int] = kwargs.get("third_kernel_size", (8, 4))

        # Initialise by calling super class
        super().__init__(in_chans=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         final_conv_length=final_conv_length, pool_mode=pool_mode,
                         second_kernel_size=second_kernel_size, third_kernel_size=third_kernel_size,
                         drop_prob=dropout_rate)


class EEGNetv4MTS(EEGNetv4, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time steps or final conv length must be specified
        time_steps: int = kwargs.get("time_steps")
        final_conv_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        dropout_rate: float = kwargs.get("dropout_rate", 0.25)
        pool_mode: str = kwargs.get("pool_mode", "mean")
        f1: int = kwargs.get("f1", 8)
        d: int = kwargs.get("d", 2)
        f2: int = kwargs.get("f2", 16)  # BrainDecode implementation suggests d*f1 instead?
        kernel_length: int = kwargs.get("kernel_length", 64)
        third_kernel_size: Tuple[int, int] = kwargs.get("third_kernel_size", (8, 4))

        # Initialise by calling super class
        super().__init__(in_chans=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         final_conv_length=final_conv_length, pool_mode=pool_mode, F1=f1, D=d, F2=f2,
                         kernel_length=kernel_length, third_kernel_size=third_kernel_size, drop_prob=dropout_rate)


class EEGResNetMTS(EEGResNet, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time steps or final pool length must be specified
        time_steps: int = kwargs.get("time_steps")
        final_pool_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        n_first_filters: int = kwargs.get("n_first_filters", 6)  # TODO: default value?
        n_layers_per_block: int = kwargs.get("n_layers_per_block", 2)
        first_filter_length: int = kwargs.get("first_filter_length", 3)
        non_linearity: Callable = kwargs.get("non_linearity", elu)
        split_first_layer: bool = kwargs.get("split_first_layer", True)
        batch_norm_alpha: float = kwargs.get("batch_norm_alpha", 0.1)
        batch_norm_epsilon: float = kwargs.get("batch_norm_epsilon", 1e-4)
        conv_weight_init_fn: Callable = kwargs.get("conv_weight_init_fn", lambda w: nn.init.kaiming_normal_(w, a=0))

        # Initialise
        super().__init__(in_chans=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         final_pool_length=final_pool_length, n_first_filters=n_first_filters,
                         n_layers_per_block=n_layers_per_block, first_filter_length=first_filter_length,
                         nonlinearity=non_linearity, split_first_layer=split_first_layer,
                         batch_norm_alpha=batch_norm_alpha, batch_norm_epsilon=batch_norm_epsilon,
                         conv_weight_init_fn=conv_weight_init_fn)


class EEGInceptionMTS(EEGInception, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # The following have defaults in BrainDecode, but they need to be passed anyway. In their documentation, it says
        # 'input_size_ms' which specifies that it is the temporal size of input in milliseconds. However, this argument
        # is not used in their __init__-method, and 'input_window_samples' is believed to have the same meaning as usual
        time_steps: int = kwargs.pop("time_steps")
        sampling_freq: Union[float, int] = kwargs.pop("sampling_freq")

        # -----------------------
        # Optional kwargs
        # -----------------------
        dropout_rate: float = kwargs.get("dropout_rate", 0.5)

        # Documentation is missing for scales_samples_s. Documentation exist for scales_time, but this does not exist in
        # input arguments of their __init__-method. Unsure if these defaults should be used for all sampling frequencies
        # and input number of time steps
        scales_samples_s: Tuple[float, float, float] = kwargs.get("scales_samples_s", (0.5, 0.25, 0.125))
        n_filters: int = kwargs.get("n_filters", 8)
        activation: nn.Module = kwargs.get("activation", nn.ELU())
        batch_norm_alpha: float = kwargs.get("batch_norm_alpha", 0.01)
        depth_multiplier: int = kwargs.get("depth_multiplier", 2)
        pooling_sizes: Tuple[int, int, int, int] = kwargs.get("pooling_sizes", (4, 2, 2, 2))

        # Initialise by calling super class
        super().__init__(in_channels=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         sfreq=sampling_freq, drop_prob=dropout_rate, scales_samples_s=scales_samples_s,
                         n_filters=n_filters, activation=activation, batch_norm_alpha=batch_norm_alpha,
                         depth_multiplier=depth_multiplier, pooling_sizes=pooling_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(super().forward(x))


class SleepStagerChambon2018MTS(SleepStagerChambon2018, BaseMTSClassifier):

    activation_function = "linear"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")
        sampling_freq: Union[float, int] = kwargs.pop("sampling_freq")
        time_steps: int = kwargs.pop("time_steps")

        input_size_s = time_steps // sampling_freq

        # -----------------------
        # Optional kwargs
        # -----------------------
        n_conv_chs: int = kwargs.get("n_conv_chs", 8)
        time_conv_size_s: float = kwargs.get("time_conv_size_s", 0.5)
        max_pool_size_s: float = kwargs.get("max_pool_size_s", 0.125)
        pad_size_s: float = kwargs.get("max_pool_size_s", 0.125)
        dropout_rate: float = kwargs.get("dropout_rate", 0.5)
        apply_batch_norm: bool = kwargs.get("apply_batch_norm", False)
        return_feats: bool = kwargs.get("return_feats", False)

        # Initialise by calling super class
        super().__init__(n_channels=in_channels, sfreq=sampling_freq, n_classes=num_classes, input_size_s=input_size_s,
                         n_conv_chs=n_conv_chs, time_conv_size_s=time_conv_size_s, max_pool_size_s=max_pool_size_s,
                         pad_size_s=pad_size_s, dropout=dropout_rate, apply_batch_norm=apply_batch_norm,
                         return_feats=return_feats)


class TIDNetMTS(TIDNet, BaseMTSClassifier):

    activation_function = "log_softmax"

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")
        time_steps: int = kwargs.pop("time_steps")

        # -----------------------
        # Optional kwargs
        # -----------------------
        s_growth: int = kwargs.get("s_growth", 24)
        t_filters: int = kwargs.get("t_filters", 32)
        dropout_rate = kwargs.get("dropout_rate", 0.4)
        pooling: int = kwargs.get("pooling", 15)
        temp_layers: int = kwargs.get("temp_layers", 2)
        spat_layers: int = kwargs.get("spat_layers", 2)
        temp_span: float = kwargs.get("temp_span", 0.05)  # Not sure if this is the best way to do it...
        bottleneck: int = kwargs.get("bottleneck", 3)
        summary: int = kwargs.get("summary", -1)

        # Initialise by calling super method
        super().__init__(in_chans=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         s_growth=s_growth, t_filters=t_filters, drop_prob=dropout_rate, pooling=pooling,
                         temp_layers=temp_layers, spat_layers=spat_layers, temp_span=temp_span, bottleneck=bottleneck,
                         summary=summary)
