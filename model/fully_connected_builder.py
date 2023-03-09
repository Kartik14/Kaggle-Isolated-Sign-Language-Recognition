from typing import Any, Tuple

import keras
import tensorflow as tf

from model.abstract_model_builder import AbstractModelBuilder


class FullyConnectedModel(AbstractModelBuilder):
    def __init__(self, input_shape: Tuple[int], **kwargs: Any):
        super().__init__(input_shape)
        self.init_fc = kwargs.get("init_fc", 512)
        self.n_blocks = kwargs.get("n_blocks", 2)
        self.dropout_1 = kwargs.get("dropout_1", 0.2)
        self.dropout_2 = kwargs.get("dropout_2", 0.6)
        self.n_labels = kwargs.get("n_labels", 250)

    @staticmethod
    def get_fc_block(inputs: tf.Tensor, output_channels: int, dropout: float = 0.2) -> tf.Tensor:
        x = keras.layers.Dense(output_channels)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.Dropout(dropout)(x)
        return x

    def create_layers(self) -> None:
        self.input_layer = keras.layers.Input(shape=self.input_shape)

        x = self.input_layer
        for i in range(self.n_blocks):
            x = self.get_fc_block(
                x,
                output_channels=self.init_fc // (2**i),
                dropout=self.dropout_1 if (1 + i) != self.n_blocks else self.dropout_2,
            )
        self.output_layer = keras.layers.Dense(self.n_labels, activation="softmax")(x)
