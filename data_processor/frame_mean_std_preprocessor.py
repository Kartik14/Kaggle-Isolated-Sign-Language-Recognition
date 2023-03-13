from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import constants


class FrameMeanStdFeatureGenV1(layers.Layer):
    """Calculates mean and std across frames for features,"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="FrameMeanStdV1")
        self.skip_z = kwargs.get("skip_z", False)
        self.landmark_ranges = [
            kwargs.get("face_idx_range", constants.FACE_LANDMARKS_RANGE),
            kwargs.get("lh_idx_range", constants.LEFT_HAND_LANDMARKS_RANGE),
            kwargs.get("pose_idx_range", constants.POSE_LANDMARKS_RANGE),
            kwargs.get("rh_idx_range", constants.RIGHT_HAND_LANDMARKS_RANGE),
        ]

        num_columns = 2 if self.skip_z else 3
        self.flat_feat_lens = [num_columns * (_r[1] - _r[0]) for _r in self.landmark_ranges]

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        if self.skip_z:
            inputs = inputs[:, :, :2]

        landmarks = [inputs[:, lrange[0] : lrange[1], :] for lrange in self.landmark_ranges]
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


class FrameMeanStdFeatureGenV2(layers.Layer):
    """Calculates mean and std across frames for features, Uses only lip features"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="FrameMeanStdV2")
        self.skip_z = kwargs.get("skip_z", False)
        self.lip_coords = kwargs.get("lips_coord", constants.LIPS_COORDINATES)
        self.landmark_ranges = [
            kwargs.get("lh_idx_range", constants.LEFT_HAND_LANDMARKS_RANGE),
            kwargs.get("pose_idx_range", constants.POSE_LANDMARKS_RANGE),
            kwargs.get("rh_idx_range", constants.RIGHT_HAND_LANDMARKS_RANGE),
        ]
        num_columns = 2 if self.skip_z else 3
        self.flat_feat_lens = [
            num_columns * len(self.lip_coords),
        ] + [num_columns * (_r[1] - _r[0]) for _r in self.landmark_ranges]

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        if self.skip_z:
            inputs = inputs[:, :, :2]

        # Get lips separately
        landmarks = [tf.gather(inputs, self.lip_coords, axis=1)]

        # Get remaining landmarks
        for lrange in self.landmark_ranges[1:]:
            landmarks.append(inputs[:, lrange[0] : lrange[1], :])

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
