import json
import os
import random
from os.path import join

import numpy as np
import pandas as pd
import tensorflow as tf

import constants


def seed_it_all(seed: int = 14) -> None:
    """Attempt to be Reproducible"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_tf_verbosity() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_sign_encoder() -> dict:
    mapping_path = join(constants.DATA_ROOT, "sign_to_prediction_index_map.json")
    with open(mapping_path) as fd:
        sign_mapping = json.load(fd)
    return sign_mapping


def get_sign_decoder() -> dict:
    sign_encoder = get_sign_encoder()
    sign_decoder = {v: k for k, v in sign_encoder.items()}
    return sign_decoder


def load_relevant_data_subset(pq_path: str, data_columns: list = ["x", "y", "z"]) -> np.ndarray:
    """reads data columns from a pandas dataframe"""
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / constants.ROWS_PER_FRAME)
    data = data.values.reshape((n_frames, constants.ROWS_PER_FRAME, len(data_columns)))
    return data.astype(np.float32)


def read_yaml_file(yaml_file: str) -> dict:
    import yaml

    with open(yaml_file) as fp:
        params = yaml.safe_load(fp)
    print(params)
    return params
