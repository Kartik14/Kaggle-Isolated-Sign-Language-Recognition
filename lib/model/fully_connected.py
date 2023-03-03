import tensorflow as tf
from tensorflow.keras import Model, layers


def get_fc_block(inputs: tf.Tensor, output_channels: int, dropout: float = 0.2) -> tf.Tensor:
    x = layers.Dense(output_channels)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_classification_model(
    n_labels: int = 250,
    init_fc: int = 512,
    n_blocks: int = 2,
    dropout_1: float = 0.2,
    dropout_2: float = 0.6,
    flat_frame_len: int = 3258,
) -> Model:
    _inputs = layers.Input(shape=(flat_frame_len,))
    x = _inputs

    # Define layers
    for i in range(n_blocks):
        x = get_fc_block(
            x, output_channels=init_fc // (2**i), dropout=dropout_1 if (1 + i) != n_blocks else dropout_2
        )

    # Define output layer
    _outputs = layers.Dense(n_labels, activation="softmax")(x)

    # Build the model
    model = Model(inputs=_inputs, outputs=_outputs)
    return model


if __name__ == "__main__":
    fc_model = build_classification_model()
    fc_model.compile(tf.keras.optimizers.Adam(0.000333), "sparse_categorical_crossentropy", metrics="acc")
    fc_model.summary()
    tf.keras.utils.plot_model(fc_model)
