import torch

from typing import List
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
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass