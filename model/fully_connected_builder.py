from typing import Any

import tensorflow as tf
from tensorflow import keras

import constants


class LinearBlock(keras.layers.Layer):
    """Linear Block with BN, Activation and Dropout"""

    def __init__(
        self,
        output_channels: int,
        activation: str = "gelu",
        dropout: float = 0.4,
        name: str = "linearBlock",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.dense = keras.layers.Dense(output_channels)
        self.bn = keras.layers.BatchNormalization()
        self.activation = keras.layers.Activation(activation)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, *args: Any, **kwargs: Any) -> tf.Tensor:
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FullyConnectedV1(keras.models.Model):
    def __init__(
        self,
        num_blocks: int = 2,
        init_fc: int = 512,
        dropout_inter: float = 0.2,
        dropout_last: float = 0.6,
        units_multiplier: float = 2.0,
        name: str = "FullyConnectedV1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.num_blocks = num_blocks
        self.init_fc = init_fc
        self.dropout_inter = dropout_inter
        self.dropout_last = dropout_last
        self.units_multiplier = units_multiplier

        self.linear_blocks = []
        for block_num in range(self.num_blocks):
            output_channels = int(self.init_fc / (self.units_multiplier**block_num))
            dropout = self.dropout_inter if (block_num + 1) != self.num_blocks else self.dropout_last
            self.linear_blocks.append(LinearBlock(output_channels, dropout=dropout, name=f"linearBlock{block_num}"))
        self.output_layer = keras.layers.Dense(constants.NUM_LABELS, activation="softmax")

    def call(self, inputs: tf.Tensor, training: Any = None, mask: Any = None) -> tf.Tensor:
        x = inputs
        for i in range(self.num_blocks):
            x = self.linear_blocks[i](x)
        x = self.output_layer(x)
        return x
