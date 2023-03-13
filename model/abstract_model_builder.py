from abc import ABC, abstractmethod
from typing import Tuple

from keras.models import Model


class AbstractModelBuilder(ABC):
    # Initialize the builder with an input shape
    def __init__(self, input_shape: Tuple[int]) -> None:
        self.input_shape = input_shape
        self.input_layer = None
        self.output_layer = None

    @abstractmethod
    def create_layers(self) -> None:
        pass

    # Build and return a Keras model using the created layers
    def build_model(self) -> Model:
        return Model(inputs=self.input_layer, outputs=self.output_layer)
