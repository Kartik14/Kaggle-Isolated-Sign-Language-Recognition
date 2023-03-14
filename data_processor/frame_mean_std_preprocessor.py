from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import constants


class FrameMeanStdFeatureGen(layers.Layer):
    def __init__(self, landmarks_set: list, **kwargs: Any) -> None:
        super().__init__(name="FrameMeanStdV1")

        self.landmarks_set = landmarks_set
        self.landmark_coords = self.get_landmark_coords()
        self.skip_z = kwargs.get("skip_z", False)

        num_columns = 2 if self.skip_z else 3
        self.flat_feat_lens = []
        for coords in self.landmark_coords:
            if isinstance(coords, tuple):
                num_coords = coords[1] - coords[0]
            elif isinstance(coords, list):
                num_coords = len(coords)
            else:
                raise TypeError("Invalid landmark coordinate type")
            self.flat_feat_lens.append(num_columns * num_coords)

    def get_landmark_coords(self) -> list:
        landmark_coords = []
        for landmark in self.landmarks_set:
            landmark_coords.append(constants.Landmarks[landmark])
        return landmark_coords

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        if self.skip_z:
            inputs = inputs[:, :, :2]

        landmarks = []
        for coords in self.landmark_coords:
            if isinstance(coords, tuple):
                landmarks.append(inputs[:, coords[0] : coords[1], :])
            elif isinstance(coords, list):
                landmarks.append(tf.gather(inputs, coords, axis=1))
            else:
                raise TypeError("Invalid landmark coordinate type")

        landmarks = [tf.reshape(_l, (-1, feat_len)) for _l, feat_len in zip(landmarks, self.flat_feat_lens)]
        landmarks = [
            tf.boolean_mask(features, tf.reduce_all(tf.logical_not(tf.math.is_nan(features)), axis=1), axis=0)
            for features in landmarks
        ]

        # Get means and stds
        feature_means = [tf.math.reduce_mean(_x, axis=0) for _x in landmarks]
        feature_stds = [tf.math.reduce_std(_x, axis=0) for _x in landmarks]

        final_feature = tf.concat([*feature_means, *feature_stds], axis=0)
        final_feature = tf.where(tf.math.is_finite(final_feature), final_feature, tf.zeros_like(final_feature))
        return final_feature
