"""
Implementing classification and regression Histories class for storing training and validation metrics during training

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
from typing import Dict, List, Optional, Union
import pickle
import warnings

import numpy
import torch
from matplotlib import pyplot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    r2_score


class Histories:
    """
    Class for keeping track of all metrics during training.
    """

    __slots__ = "_history", "_epoch_y_pred", "_epoch_y_true"

    classification_metrics = "accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "mcc", "auc"
    regression_metrics = "mse", "mae", "rmse", "mape", "pearson_cc", "spearman_cc"  # Not using R^2 as default due to
    # non-linear regression

    def __init__(self, metrics: Optional[Union[str, List[str]]] = None):
        """
        Initialise
        Args:
            metrics: A list of metrics to use. If set to None, all available metrics will be used
        Examples:
            >>> my_history = Histories(metrics=["accuracy", "sensitivity"])
            >>> sorted(my_history.history.items())
            [('accuracy', []), ('sensitivity', [])]
            >>> my_history = Histories(metrics=["accuracy", "recall", "matthews_correlation"])  # implemented as mcc
            >>> sorted(my_history.history.items())
            [('accuracy', []), ('recall', [])]
            >>> sorted(Histories().history.keys())
            ['accuracy', 'auc', 'f1', 'mcc', 'precision', 'recall', 'sensitivity', 'specificity']
            >>> _ = Histories(metrics=["acc", "rec"])  # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
            ...
            ValueError: No implemented metrics were found. Please pass at least one of the following: ('accuracy',
                'sensitivity', 'specificity', 'precision', 'recall', 'f1', 'mcc', 'auc', 'mse', 'mae', 'rmse', 'mape',
                'pearson_cc', 'spearman_cc')
        """

        # Give warning for all metrics not included (not in legal_metrics)
        legal_metrics = self.classification_metrics + self.regression_metrics
        if metrics == "regression":
            metrics = self.regression_metrics
        elif metrics in ("classification", None):
            metrics = self.classification_metrics if metrics is None else metrics
        for metric in metrics:
            if metric not in legal_metrics:
                warnings.warn(f"The metric {metric} is not supported and will be ignored")

        # Keep only the supported ones
        metrics = list(set(metrics) & set(legal_metrics))
        if not metrics:
            raise ValueError(f"No implemented metrics were found. Please pass at least one of the following: "
                             f"{legal_metrics}")

        # Create history dictionary
        self._history: Dict[str, List[float]] = {f"{metric}": [] for metric in metrics}

        # Initialise epochs predictions and targets. They will be updated for each batch
        self._epoch_y_pred: List[torch.Tensor] = []
        self._epoch_y_true: List[torch.Tensor] = []

    def store_batch_evaluation(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Store the predictions and targets. Should be called for each batch"""
        self._epoch_y_pred.append(y_pred)
        self._epoch_y_true.append(y_true)

    def on_epoch_end(self, verbose: bool = True) -> None:
        """
        Updates and print newest metrics. Use this after each epoch
        Args:
            verbose: To print the newly calculated metrics or not.
        Returns: Nothing

        Examples:
            >>> my_hist = Histories(metrics=["accuracy", "sensitivity", "f1", "mcc"])
            >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78])
            >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1.])
            >>> my_hist.store_batch_evaluation(y_pred=my_pred[:4], y_true=my_true[:4])  # First batch
            >>> my_hist.store_batch_evaluation(y_pred=my_pred[4:], y_true=my_true[4:])  # Second batch
            >>> my_hist.on_epoch_end(verbose=False)  # Epoch done
            >>> sorted(my_hist.history.items())  # doctest: +NORMALIZE_WHITESPACE
            [('accuracy', [0.7142857142857143]), ('f1', [0.7499999999999999]), ('mcc', [0.5477225575051661]),
             ('sensitivity', [0.6])]

        """
        self._update_metrics()
        if verbose:
            self._print_newest_metrics()

    def _update_metrics(self) -> None:
        """
        Updates all metric histories based on the predictions and targets stored after each batch.
        Returns: Nothing, just updates metric histories

        Examples:
            >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78])
            >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1.])
            >>> my_hist = Histories(metrics=["accuracy", "sensitivity", "f1", "mcc"])
            >>> my_hist.store_batch_evaluation(y_pred=my_pred[:4], y_true=my_true[:4])  # First batch_size = 4
            >>> sorted(my_hist.history.items()) # Histories not yet updated
            [('accuracy', []), ('f1', []), ('mcc', []), ('sensitivity', [])]
            >>> my_hist.store_batch_evaluation(y_pred=my_pred[4:], y_true=my_true[4:])  # Second batch_size = 3
            >>> my_hist._update_metrics()  # Update metrics
            >>> sorted(my_hist.history.items())  # doctest: +NORMALIZE_WHITESPACE
            [('accuracy', [0.7142857142857143]), ('f1', [0.7499999999999999]), ('mcc', [0.5477225575051661]),
             ('sensitivity', [0.6])]
        """
        # Update all metrics
        for metric, hist in self._history.items():
            hist.append(Histories._compute_metric(y_pred=torch.cat(self._epoch_y_pred, dim=0),
                                                  y_true=torch.cat(self._epoch_y_true, dim=0), metric=metric))

        # Remove the epoch histories
        self._epoch_y_pred = []
        self._epoch_y_true = []

    def _print_newest_metrics(self) -> None:
        for i, (metric_name, metric_values) in enumerate(self.history.items()):
            if i == len(self.history) - 1:
                print(f"{metric_name}: {metric_values[-1]:.3f}")
            else:
                print(f"{metric_name}: {metric_values[-1]:.3f}\t\t", end="")

    @staticmethod
    def _compute_metric(y_pred: torch.Tensor, y_true: torch.Tensor, metric: str) -> float:
        if metric == "accuracy":
            score = _accuracy(y_pred=y_pred, y_true=y_true)
        elif metric == "sensitivity":
            score = _sensitivity(y_pred=y_pred, y_true=y_true)
        elif metric == "specificity":
            score = _specificity(y_pred=y_pred, y_true=y_true)
        elif metric == "precision":
            score = _precision(y_pred=y_pred, y_true=y_true)
        elif metric == "recall":
            score = _recall(y_pred=y_pred, y_true=y_true)
        elif metric == "f1":
            score = _f1_score(y_pred=y_pred, y_true=y_true)
        elif metric == "mcc":
            score = _matthews_cc(y_pred=y_pred, y_true=y_true)
        elif metric == "auc":
            score = _area_under_roc(y_pred=y_pred, y_true=y_true)
        elif metric == "mse":
            score = _mse(y_pred=y_pred, y_true=y_true)
        elif metric == "mae":
            score = _mae(y_pred=y_pred, y_true=y_true)
        elif metric == "rmse":
            score = _rmse(y_pred=y_pred, y_true=y_true)
        elif metric == "mape":
            score = _mape(y_pred=y_pred, y_true=y_true)
        elif metric == "r2":
            score = _r2_score(y_pred=y_pred, y_true=y_true)
        elif metric == "pearson_cc":
            score = _pearson_cc(y_pred=y_pred, y_true=y_true)
        elif metric == "spearman_cc":
            score = _spearman_cc(y_pred=y_pred, y_true=y_true)
        else:
            raise NotImplementedError(f"The metric {metric} has not been implemented for this method")
        return score

    # -----------------------------------------
    # Properties
    # -----------------------------------------
    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history

    @property
    def newest_metrics(self) -> Dict[str, float]:
        """Returns the newest calculated metric"""
        return {metric_name: metric_values[-1] for metric_name, metric_values in self._history.items()}


def save_all_histories(histories: Dict[str, Histories], path: str) -> None:
    """
    Saves the metric curves as images and pickle files
    Args:
        histories: A Dict with keys being names, used to create names of pickle file and label in plot
        path: path to save the plots
    Returns: Nothing, it just saves the performance, both as plots and as dictionaries
    """
    # --------------------------------
    # Check and get metrics
    # --------------------------------
    metrics = list(histories.values())[0].history.keys()

    # --------------------------------
    # Plot and save all metrics curves
    # --------------------------------
    for i, metric in enumerate(metrics):
        # Plot
        pyplot.figure()

        for name, history in histories.items():
            pyplot.plot(history.history[metric], label=name)

        pyplot.title(f"Metric: {metric}")
        pyplot.legend()

        # Save plots to folder
        pyplot.savefig(f"{path}/{metric}.png")

    # --------------------------------
    # Save histories object as pickle file
    # --------------------------------
    for name, history in histories.items():
        with open(f"{path}/{name.lower()}_history.pkl", "wb") as f:
            pickle.dump(history.history, f)


# -----------------------------------------
# Regression metrics
# -----------------------------------------
def _mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate mean squared error
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: MSE
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> _mse(y_pred=my_pred, y_true=my_true)  # should equal 57/16
        3.5625
    """
    return mean_squared_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())


def _rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate root mean squared error
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: RMSE
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> round(_rmse(y_pred=my_pred, y_true=my_true), 5)  # should equal sqrt(57/16)
        1.88746
    """
    return numpy.sqrt(mean_squared_error(y_true=y_true.cpu(), y_pred=y_pred.cpu()))


def _mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate mean absolute error
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: MAE
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> _mae(y_pred=my_pred, y_true=my_true)  # should equal 6.5/4
        1.625
    """
    return mean_absolute_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())


def _mape(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate mean absolute percentage error
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: MAPE
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> _mape(y_pred=my_pred, y_true=my_true)
        0.2683982683982684
    """
    return mean_absolute_percentage_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())


def _r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate R squared
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: R squared
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> round(_r2_score(y_pred=my_pred, y_true=my_true), 5)
        0.69801
    """
    return r2_score(y_true=y_true.cpu(), y_pred=y_pred.cpu())


def _pearson_cc(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Pearson's correlation coefficient (Pearson's r)
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: Pearson CC
    Examples:
        >>> my_pred = torch.unsqueeze(torch.tensor([9, 4, 8, 14]), dim=1)
        >>> my_true = torch.unsqueeze(torch.tensor([7, 6, 5.5, 14]), dim=1)
        >>> _pearson_cc(y_pred=my_pred, y_true=my_true)
        0.8710295406991435
    """
    # Removing redundant dimension may be necessary
    if y_true.dim() == 2:
        y_true = torch.squeeze(y_true, dim=1)
    if y_pred.dim() == 2:
        y_pred = torch.squeeze(y_pred, dim=1)

    # Compute and return
    return pearsonr(x=y_true.cpu(), y=y_pred.cpu())[0]


def _spearman_cc(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Spearman's correlation coefficient (Spearman's rho)
    Args:
        y_pred: predictions with shape=(batch,) or shape=(batch, outputs)
        y_true: targets with shape=(batch,) or shape=(batch, outputs)

    Returns: Spearman's rho
    Examples:
        >>> my_pred = torch.tensor([9, 4, 8, 14])
        >>> my_true = torch.tensor([7, 6, 5.5, 14])
        >>> round(_spearman_cc(y_pred=my_pred, y_true=my_true), 5)
        0.8
    """
    # Removing redundant dimension may be necessary
    if y_true.dim() == 2:
        y_true = torch.squeeze(y_true, dim=1)
    if y_pred.dim() == 2:
        y_pred = torch.squeeze(y_pred, dim=1)

    # Compute and return
    return spearmanr(a=y_true.cpu(), b=y_pred.cpu())[0]


# -----------------------------------------
# TP, FP, TN, FN
# -----------------------------------------
def _true_positives(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate number of true positives
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _true_positives(y_pred=my_pred, y_true=my_true)
        4
    """
    # Make predictions binary
    y_pred = torch.round(y_pred)

    return sum([1 for prediction, target in zip(y_pred, y_true) if int(prediction) == 1 and int(target) == 1])


def _false_positives(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate number of false positives
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _false_positives(y_pred=my_pred, y_true=my_true)
        1
    """
    # Make predictions binary
    y_pred = torch.round(y_pred)

    return sum([1 for prediction, target in zip(y_pred, y_true) if int(prediction) == 1 and int(target) == 0])


def _true_negatives(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate number of true negatives
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _true_negatives(y_pred=my_pred, y_true=my_true)
        2
    """
    # Make predictions binary
    y_pred = torch.round(y_pred)

    return sum([1 for prediction, target in zip(y_pred, y_true) if int(prediction) == 0 and int(target) == 0])


def _false_negatives(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate number of false negatives
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _false_negatives(y_pred=my_pred, y_true=my_true)
        3
    """
    # Make predictions binary
    y_pred = torch.round(y_pred)

    return sum([1 for prediction, target in zip(y_pred, y_true) if int(prediction) == 0 and int(target) == 1])


# -----------------------------------------
# Classification metrics
# -----------------------------------------
def _accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates accuracy
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch)
    Returns: Accuracy

    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _accuracy(y_pred=my_pred, y_true=my_true)
        0.6
    """

    tp = _true_positives(y_pred=y_pred, y_true=y_true)
    tn = _true_negatives(y_pred=y_pred, y_true=y_true)
    fp = _false_positives(y_pred=y_pred, y_true=y_true)
    fn = _false_negatives(y_pred=y_pred, y_true=y_true)

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    return (tp + tn) / numpy.sum((tp, tn, fp, fn))


def _sensitivity(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate sensitivity
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)
    Returns: Sensitivity

    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _sensitivity(y_pred=my_pred, y_true=my_true)
        0.5714285714285714
    """
    tp = _true_positives(y_pred=y_pred, y_true=y_true)
    fn = _false_negatives(y_pred=y_pred, y_true=y_true)

    # Sensitivity = TP / (TP + FN)
    return tp / numpy.sum((tp, fn))


def _specificity(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates specificity
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)
    Returns: Specificity
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _specificity(y_pred=my_pred, y_true=my_true)
        0.6666666666666666
    """
    tn = _true_negatives(y_pred=y_pred, y_true=y_true)
    fp = _false_positives(y_pred=y_pred, y_true=y_true)

    # Specificity = TN / (TN + FP)
    return tn / numpy.sum((tn, fp))


def _precision(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates precision
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)
    Returns: Precision
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _precision(y_pred=my_pred, y_true=my_true)
        0.8
    """
    tp = _true_positives(y_pred=y_pred, y_true=y_true)
    fp = _false_positives(y_pred=y_pred, y_true=y_true)

    # Precision = TP / (TP + FP)
    return tp / numpy.sum((tp, fp))


def _recall(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate recall
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)
    Returns: Sensitivity
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _recall(y_pred=my_pred, y_true=my_true)
        0.5714285714285714
    """
    tp = _true_positives(y_pred=y_pred, y_true=y_true)
    fn = _false_negatives(y_pred=y_pred, y_true=y_true)

    # Recall = TP / (TP + FN)
    return tp / numpy.sum((tp, fn))


def _f1_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate F1 score
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)
    Returns: F1 score
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _f1_score(y_pred=my_pred, y_true=my_true)
        0.6666666666666666
    """
    precision = _precision(y_pred=y_pred, y_true=y_true)
    recall = _recall(y_pred=y_pred, y_true=y_true)

    # F1 = 2 * precision * recall / (precision + recall)
    return 2 * precision * recall / numpy.sum((precision, recall))


def _matthews_cc(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates Matthews Correlation Coefficient. nan values may occur, as MCC sometimes get a denominator equal to 0
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)

    Returns:
        Matthews Correlation Coefficient
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _matthews_cc(y_pred=my_pred, y_true=my_true)
        0.2182178902359924
        >>> my_pred = torch.tensor([.0, .0, .0, .0, .0, .0, .0, .0, .0, .0])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _matthews_cc(y_pred=my_pred, y_true=my_true)
        nan
    """
    tp = _true_positives(y_pred=y_pred, y_true=y_true)
    tn = _true_negatives(y_pred=y_pred, y_true=y_true)
    fp = _false_positives(y_pred=y_pred, y_true=y_true)
    fn = _false_negatives(y_pred=y_pred, y_true=y_true)

    # MCC = (TN * TP - FN * FP) / sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FN))
    return (tn * tp - fn * fp) / numpy.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def _area_under_roc(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate AUC
    Args:
        y_pred: predictions, with shape=(batch,)
        y_true: targets, with shape=(batch,)

    Returns:
        AUC score
    Examples:
        >>> my_pred = torch.tensor([.3, .7, .2, .8, .2, .2, .78, .9, .87, .3])
        >>> my_true = torch.tensor([1., 1., 0., 1., 0., 1., 1., .0, 1., 1.])
        >>> _area_under_roc(y_pred=my_pred, y_true=my_true)
        0.6190476190476192
    """
    return roc_auc_score(y_true=y_true.cpu(), y_score=y_pred.cpu())
