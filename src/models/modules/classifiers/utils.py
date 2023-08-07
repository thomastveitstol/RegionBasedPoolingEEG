from typing import Type

import torch.nn as nn


def get_loss(activation_function: str, num_classes: int) -> Type:

    if activation_function == "log_softmax":
        return nn.NLLLoss
    elif activation_function == "linear":
        if num_classes > 1:
            return nn.CrossEntropyLoss
        elif num_classes == 1:
            return nn.BCEWithLogitsLoss
    else:
        raise ValueError(f"Combination of activation function {activation_function} and num classes {num_classes} "
                         f"was not recognised")
