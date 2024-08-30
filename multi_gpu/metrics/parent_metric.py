import torch
import numpy as np

from typing import List, Tuple
from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Abstract base class for metrics.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state.
        """
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        """
        pass

    @abstractmethod
    def calculate_metric(self) -> float:
        """
        Compute and return the metric.
        """
        pass
    
    
    
class MetricTracker():
    
    """
    A class to track multiple metrics.
    """
    
    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics = metrics

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        for metric in self.metrics:
            metric.update(prediction, target)

    def calculate_metric(self, local_rank : int) -> List[float]:
        return [metric.calculate_metric(local_rank) for metric in self.metrics]