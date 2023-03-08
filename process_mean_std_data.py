import warnings
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
from helper import utils
from helper.logging import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

ranges = [
    constants.FACE_LANDMARKS_RANGE,
    constants.LEFT_HAND_LANDMARKS_RANGE,
    constants.POSE_LANDMARKS_RANGE,
    constants.RIGHT_HAND_LANDMARKS_RANGE,
]


def get_mean_std_feat(landmark_coords: np.ndarray) -> np.ndarray:
    # separate into each landmark
    num_frames = landmark_coords.shape[0]
    features = [
        landmark_coords[:, landmark_range[0] : landmark_range[1], :].reshape(num_frames, -1)
        for landmark_range in ranges
    ]

    # remove nan values from feature landmarks
    features = [feature[np.all(np.logical_not(np.isnan(feature)), axis=1)] for feature in features]

    # Extract mean and std information
    features_mean = [np.mean(feature, axis=0) for feature in features]
    features_std = [np.std(feature, axis=0) for feature in features]
    final_feature = np.concatenate([*features_mean, *features_std])
    final_feature = np.where(np.isfinite(final_feature), final_feature, np.zeros_like(final_feature))
    return final_feature


def process() -> None:
    # read train json
    train_df = pd.read_csv(join(constants.DATA_ROOT, "train.csv"))

    # get sign to prediction index mapping
    sign_to_prediction_index = utils.get_sign_encoder()

    # only using 'x' and 'y' data columns
    data_columns = ["x", "y", "z"]

    # iterate over each row in the frame
    landmark_features_array = []
    landmark_labels_array = []
    for row_id, row in tqdm(train_df.iterrows(), total=len(train_df)):
        sample_path = row["path"]
        sample_data = pd.read_parquet(join(constants.DATA_ROOT, sample_path))
        n_frames = len(sample_data) // constants.ROWS_PER_FRAME
        landmark_data = (
            sample_data[data_columns].to_numpy().reshape(n_frames, constants.ROWS_PER_FRAME, len(data_columns))
        )
        landmark_features = get_mean_std_feat(landmark_data)
        if len(landmark_features) == 0:
            logger.warning(f"Invalid feature {row_id}")
            continue
        landmark_features_array.append(landmark_features)

        landmark_label = row["sign"]
        landmark_labels_array.append(sign_to_prediction_index[landmark_label])

    # save paths
    data_save_path = join(constants.DATA_ROOT, "feature_data_nonan.npy")
    np.save(data_save_path, landmark_features_array)
    labels_save_path = join(constants.DATA_ROOT, "labels_data.npy")
    np.save(labels_save_path, landmark_labels_array)


if __name__ == "__main__":
    process()
