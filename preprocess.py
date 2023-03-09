from os.path import join
from parser.preprocess_parser import PreprocessParser
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
from data_processor.abstract_preprocessor import AbstractDataProcessor
from data_processor.frame_mean_std_preprocessor import FrameMeanStdPreprocessor
from helper.logging import logger
from helper.utils import get_sign_encoder, load_relevant_data_subset_with_imputation


class Preprocess:
    """Class for preprocessing data for training"""

    def __init__(self) -> None:
        self.params = PreprocessParser().parse_args()
        self.preprocessor = self.get_preprocessor()
        self.preprocessed_data: Dict[str, list] = {"features": list(), "labels": list()}

        self.train_df = pd.read_csv(self.params.train_csv)
        if self.params.expt_run:
            self.train_df = self.train_df.head(100)
        self.sign_to_prediction_index = get_sign_encoder()

    def get_preprocessor(self) -> AbstractDataProcessor:
        if self.params.mode == "frame_mean_std":
            return FrameMeanStdPreprocessor(self.params.use_z)
        else:
            raise RuntimeError

    def process(self) -> None:
        """extract features from each data sample"""

        for row_id, row in tqdm(self.train_df.iterrows(), total=len(self.train_df)):
            pq_path, label = row["path"], row["sign"]
            prediction_index = self.sign_to_prediction_index[label]
            landmark_features = load_relevant_data_subset_with_imputation(join(constants.DATA_ROOT, pq_path))
            processed_features = self.preprocessor.transform(landmark_features)
            self.preprocessed_data["features"].append(processed_features)
            self.preprocessed_data["labels"].append(prediction_index)

        self.save_data()

    def save_data(self) -> None:
        logger.info(f"Saving preprocessed data to {self.params.output_path}")
        np.save(self.params.output_path, self.preprocessed_data)


if __name__ == "__main__":
    Preprocess().process()
