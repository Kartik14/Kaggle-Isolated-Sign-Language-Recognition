import multiprocessing as mp
import os
from os.path import join
from parser.preprocess_parser import PreprocessParser
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import constants
from data_processor.frame_mean_std_preprocessor import (
    FrameMeanStdFeatureGenV1,
    FrameMeanStdFeatureGenV2,
)
from helper.logging import logger
from helper.utils import (
    get_sign_encoder,
    load_relevant_data_subset,
    seed_it_all,
    set_tf_verbosity,
)


class Preprocess:
    """Class for preprocessing data for training"""

    def __init__(self) -> None:
        self.params = PreprocessParser().parse_args()
        self.preprocessor = self.get_preprocessor()
        self.preprocessed_data: Dict[str, list] = {
            "features": list(),
            "labels": list(),
            "participant_id": list(),
        }

        self.train_df = pd.read_csv(self.params.train_csv)
        if self.params.expt_run:
            self.train_df = self.train_df.head(1000)
        self.train_df["label"] = self.train_df["sign"].map(get_sign_encoder())

    def get_preprocessor(self) -> tf.keras.layers.Layer:
        if self.params.mode == "frame_mean_std_v1":
            return FrameMeanStdFeatureGenV1(**vars(self.params))
        elif self.params.mode == "frame_mean_std_v2":
            return FrameMeanStdFeatureGenV2(**vars(self.params))
        else:
            raise RuntimeError(f"Invalid Preprocessor type {self.params.mode}")

    def convert_features(self, pq_path: str) -> tf.Tensor:
        feature_data = load_relevant_data_subset(join(constants.DATA_ROOT, pq_path))
        return self.preprocessor(feature_data).numpy()

    def process(self) -> None:
        """extract features from each data sample"""

        with mp.Pool(processes=12, initializer=set_tf_verbosity) as pool:
            preprocessed_results = pool.imap(self.convert_features, self.train_df.path.tolist(), chunksize=250)
            for feature, (_, meta_data) in tqdm(
                zip(preprocessed_results, self.train_df.iterrows()), total=len(self.train_df)
            ):
                self.preprocessed_data["features"].append(feature)
                self.preprocessed_data["labels"].append(meta_data["label"])
                self.preprocessed_data["participant_id"].append(meta_data["participant_id"])

        self.save_data()

    def save_data(self) -> None:
        logger.info(f"Saving preprocessed data to {self.params.output_path}")
        np.save(self.params.output_path, self.preprocessed_data)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    seed_it_all()
    Preprocess().process()
