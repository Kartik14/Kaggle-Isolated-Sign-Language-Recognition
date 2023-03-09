import json
import os
import random
from os.path import join
from parser.train_parser import TrainParser
from pprint import pprint
from typing import Tuple

import keras
import numpy as np
from keras.models import Model

import constants
from helper.logging import logger
from helper.utils import get_sign_encoder
from model.fully_connected_builder import FullyConnectedModel


class Trainer:
    """Class for training models"""

    def __init__(self) -> None:
        self.params = TrainParser().parse_args()

        # create output directory
        self.result_dir = join(constants.OUTPUT_ROOT, self.params.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)

        # save params
        pprint(vars(self.params))
        logger.info(f'Saving train params to file {join(self.result_dir, "params.json")}')
        with open(join(self.result_dir, "params.json"), "w") as param_fp:
            json.dump(vars(self.params), param_fp, indent=2)

        # load data
        self.features, self.labels = self.get_dataset()
        self.train_x, self.train_y, self.val_x, self.val_y = self.split_train_validation()
        self.input_shape = self.train_x[0].shape
        self.sign_to_prediction_id = get_sign_encoder()
        self.num_labels = len(self.sign_to_prediction_id)

        # get model
        self.model_architecture = self.params.model_architecture
        self.model = self.get_model()
        self.optimizer = keras.optimizers.Adam(self.params.lr)
        self.loss_function = "sparse_categorical_crossentropy"
        self.model.compile(self.optimizer, self.loss_function, metrics="acc")
        self.model.summary()

        # train
        self.callback_list = self.get_callbacks()

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels from a numpy file."""
        dataset = np.load(self.params.train_data, allow_pickle=True).item()
        features = np.array(dataset["features"], dtype=np.float32)
        labels = np.array(dataset["labels"], dtype=np.uint8)
        assert (
            features.shape[0] == labels.shape[0]
        ), f"features {features.shape[0]} and labels array len mismatch {labels.shape[0]}"
        return features, labels

    def split_train_validation(self) -> Tuple[np.ndarray, ...]:
        """Split features and labels into train and validation sets."""
        validation_fraction = self.params.validation_fraction

        total_samples = len(self.features)
        num_validation = int(total_samples * validation_fraction)
        num_train = total_samples - num_validation

        data_idxs = list(range(total_samples))
        random.shuffle(data_idxs)
        train_idxs, val_idxs = data_idxs[:num_train], data_idxs[num_train:]

        val_x, val_y = self.features[val_idxs], self.labels[val_idxs]
        train_x, train_y = self.features[train_idxs], self.labels[train_idxs]

        return train_x, train_y, val_x, val_y

    def get_model(self) -> Model:
        """defines and build model layers"""
        if self.model_architecture == "fully_connected":
            model_obj = FullyConnectedModel(input_shape=self.input_shape, n_labels=self.num_labels)
            model_obj.create_layers()
            return model_obj.build_model()
        else:
            raise RuntimeError

    def get_callbacks(self) -> list:
        """returns callbacks for training"""
        return [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, verbose=1),
            keras.callbacks.CSVLogger(join(self.result_dir, "training_log.csv")),
        ]

    def train(self) -> None:
        self.model.fit(
            self.train_x,
            self.train_y,
            validation_data=(self.val_x, self.val_y),
            epochs=self.params.epoch,
            callbacks=self.callback_list,
            batch_size=self.params.batch_size,
        )

        # save model
        logger.info(f"Saving best trained model to folder {self.result_dir}")
        self.model.save(self.result_dir)


if __name__ == "__main__":
    Trainer().train()
