from os.path import join
from parser.tflite_parser import TfLiteParser
from typing import Any, Dict

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import constants
from data_processor.frame_mean_std_preprocessor import (
    FrameMeanStdFeatureGenV1,
    FrameMeanStdFeatureGenV2,
)
from helper.logging import logger
from helper.utils import get_sign_decoder, load_relevant_data_subset


class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model
    """

    def __init__(self, input_layer: keras.layers.Layer, islr_models: list, **kwargs: Any) -> None:
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super().__init__()

        # Load the feature generation and main models
        self.prep_inputs = input_layer
        self.islr_models = islr_models
        self.distribution_mean = tf.constant(kwargs.get("distribution_mean", 0.0), dtype=tf.float32)
        self.distribution_std = tf.constant(kwargs.get("distribution_std", 1.0), dtype=tf.float32)

    def standardize_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.where(tensor != 0, (tensor - self.distribution_mean) / self.distribution_std, tf.zeros_like(tensor))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")])
    def __call__(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = tf.expand_dims(self.prep_inputs(tf.cast(inputs, dtype=tf.float32)), axis=0)
        x = self.standardize_tensor(x)
        outputs = tf.reduce_mean(tf.concat([islr_model(x) for islr_model in self.islr_models], axis=0), axis=0)

        # Return a dictionary with the output tensor
        return {"outputs": outputs}


def save_tflite_model(keras_model: tf.Module, save_path: str) -> None:
    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = keras_model_converter.convert()
    with open(join(save_path), "wb") as f:
        f.write(tflite_model)


def get_input_layer(input_mode: str) -> keras.layers.Layer:
    if input_mode == "frame_mean_std_v1":
        return FrameMeanStdFeatureGenV1()
    elif input_mode == "frame_mean_std_v2":
        return FrameMeanStdFeatureGenV2()
    else:
        raise TypeError("Invalid Input layer mode")


if __name__ == "__main__":
    params = TfLiteParser().parse_args()

    model_kwargs = dict()

    # load data statistics for standardisation
    if params.norm_stats is not None:
        data_stats = np.load(params.norm_stats, allow_pickle=True).item()
        model_kwargs["distribution_mean"] = data_stats["distribution_mean"]
        model_kwargs["distribution_std"] = data_stats["distribution_std"]

    model_dirs = params.model_dirs
    models = []
    for model_dir in model_dirs:
        model = keras.models.load_model(model_dir)
        models.append(model)

    print(params.input_mode)
    input_layer = get_input_layer(params.input_mode)
    tflite_keras_model = TFLiteModel(input_layer, models, **model_kwargs)

    # predict on sample
    decoder = get_sign_decoder()
    train_df = pd.read_csv(join(constants.DATA_ROOT, "train.csv"))
    for _, test_sample in train_df.sample(20).iterrows():
        pq_path = test_sample["path"]
        sign = test_sample["sign"]
        demo_output = tflite_keras_model(load_relevant_data_subset(join(constants.DATA_ROOT, pq_path)))
        print(f"Prediction: {decoder[np.argmax(demo_output['outputs'])]}, Label: {sign}")

    tflite_model_path = join(params.save_dir, "model.tflite")
    save_tflite_model(tflite_keras_model, tflite_model_path)
    logger.info(f"Saved tflite model {tflite_model_path}")
