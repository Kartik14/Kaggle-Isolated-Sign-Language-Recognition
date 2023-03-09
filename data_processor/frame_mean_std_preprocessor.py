import numpy as np

from data_processor.abstract_preprocessor import AbstractDataProcessor


class FrameMeanStdPreprocessor(AbstractDataProcessor):
    def __init__(self, use_z: bool = True):
        super().__init__(use_z)

    def transform(self, landmark_features: np.ndarray) -> np.ndarray:
        if not self.use_z:
            landmark_features = landmark_features[:, :, :2]

        num_frames = landmark_features.shape[0]
        landmark_features = landmark_features.reshape(num_frames, -1)
        features_mean = np.mean(landmark_features, axis=0)
        features_std = np.std(landmark_features, axis=0)
        final_feature = np.concatenate([features_mean, features_std])
        final_feature = np.where(np.isfinite(final_feature), final_feature, np.zeros_like(final_feature))
        return final_feature
