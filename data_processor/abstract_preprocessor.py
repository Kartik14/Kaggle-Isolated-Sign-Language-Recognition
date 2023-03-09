from abc import ABC, abstractmethod

import numpy as np


class AbstractDataProcessor(ABC):
    """Abstract class for data processors"""

    def __init__(self, use_z: bool) -> None:
        self.use_z = use_z

    @abstractmethod
    def transform(self, landmark_features: np.ndarray) -> np.ndarray:
        """Transforms the raw landmark features data for training"""
        pass
