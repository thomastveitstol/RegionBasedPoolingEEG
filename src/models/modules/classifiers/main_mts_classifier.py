"""
For using all models
"""
from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.modules.classifiers.base_classifiers import BaseMTSClassifier
from src.models.modules.classifiers.mts_classifiers.inception_time import Inception
from src.models.modules.classifiers.mts_classifiers.brain_decode_models import EEGITNetMTS, Deep4NetMTS, EEGNetv1MTS, \
    EEGNetv4MTS, EEGResNetMTS, EEGInceptionMTS, SleepStagerChambon2018MTS, TIDNetMTS


class MTSClassifier(nn.Module):
    """
    Main class for Multivariate time series classifiers
    """

    def __init__(self, classifier_name: str, **kwargs):
        """
        Initialise
        Args:
            classifier_id: Name of classifier to use
            **kwargs: Hyperparameters of the classifier, which depends on the selected model
        Examples:
            >>> _ = MTSClassifier("Inception", in_channels=64, num_classes=7)
            >>> _ = MTSClassifier("EEGITNet", in_channels=64, num_classes=7, time_steps=1500)
            >>> _ = MTSClassifier("Deep4Net", in_channels=64, num_classes=7, time_steps=1500)
            >>> _ = MTSClassifier("EEGNetv1", in_channels=64, num_classes=7, time_steps=1500)
            >>> _ = MTSClassifier("EEGNetv4", in_channels=64, num_classes=7, time_steps=1500)
            >>> _ = MTSClassifier("EEGResNet", in_channels=64, num_classes=7, time_steps=1500)
            >>> _ = MTSClassifier("EEGInception", in_channels=64, num_classes=7, time_steps=1500, sampling_freq=500)
            >>> _ = MTSClassifier("SleepStagerChambon2018", in_channels=64, num_classes=7, sampling_freq=500,
            ...                time_steps=15000)  # Needs long input sequence
            >>> _ = MTSClassifier("TIDNet", in_channels=64, num_classes=7, time_steps=1500)
            >>> # Raises KeyError if classifier is not found
            >>> _ = MTSClassifier(classifier_name="ThisIsNotARealClassifier", in_channels=40)
            Traceback (most recent call last):
            ...
            KeyError: 'Classifier ThisIsNotARealClassifier was not recognised as an MTS classifier'
        """
        super().__init__()

        self._classifier = self._get_classifier(classifier_name, **kwargs)
        self._name = classifier_name

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward method

        (unittests in test folder)
        Args:
            x: EEG data, with shape=(batch, channels, time_steps)

        Returns:
            Output of the selected classifier
        """
        return self._classifier(x, **kwargs)

    @staticmethod
    def _get_classifier(classifier_name, **kwargs) -> BaseMTSClassifier:
        if classifier_name == "Inception":
            return Inception(**kwargs)
        elif classifier_name in ("EEGITNet", "EEGITNetMTS"):
            return EEGITNetMTS(**kwargs)
        elif classifier_name in ("Deep4Net", "Deep4NetMTS"):
            return Deep4NetMTS(**kwargs)
        elif classifier_name in ("EEGNetv1", "EEGNetv1MTS"):
            return EEGNetv1MTS(**kwargs)
        elif classifier_name in ("EEGNetv4", "EEGNetv4MTS"):
            return EEGNetv4MTS(**kwargs)
        elif classifier_name in ("EEGResNet", "EEGResNetMTS"):
            return EEGResNetMTS(**kwargs)
        elif classifier_name in ("EEGInception", "EEGInceptionMTS"):
            return EEGInceptionMTS(**kwargs)
        elif classifier_name in ("SleepStagerChambon2018", "SleepStagerChambon2018MTS"):
            return SleepStagerChambon2018MTS(**kwargs)
        elif classifier_name in ("TIDNet", "TIDNetMTS"):
            return TIDNetMTS(**kwargs)
        else:
            raise KeyError(f"Classifier {classifier_name} was not recognised as an MTS classifier")

    # ----------------
    # Save and load
    # ----------------
    def save(self, path: str) -> None:
        """
        Method for saving
        Args:
            path: Path to save object to

        Returns: Nothing

        """
        # Get state (everything needed to load the model)
        state = {"state_dict": self.state_dict(), "classifier_name": self._name,
                 "hyperparameters": self.hyperparameters}

        # Save
        torch.save(state, f"{path}")

    @classmethod
    def from_disk(cls, path: str) -> 'MTSClassifier':
        # Get state
        state = torch.load(path)

        # Initialise model
        model = cls(classifier_name=state["classifier_name"], **state["hyperparameters"])

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model

    # ---------------------
    # Properties
    # ---------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self._classifier.hyperparameters

    @property
    def final_activation(self) -> str:
        return self._classifier.activation_function
