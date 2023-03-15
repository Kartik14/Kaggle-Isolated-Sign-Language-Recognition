import gc
import json
import os
from os.path import join
from parser.train_parser import TrainParser
from pprint import pprint
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras
from tensorflow.keras.models import Model

import constants
from helper.logging import logger
from helper.utils import get_sign_encoder, read_yaml_file, seed_it_all
from model.fully_connected_builder import FullyConnectedV1


class Trainer:
    """Class for training models"""

    def __init__(self) -> None:
        self.runtime_params = TrainParser().parse_args()
        self.model_params = read_yaml_file(self.runtime_params.model_config)
        self.model_architecture = self.model_params["type"]
        self.all_params = {**vars(self.runtime_params), **self.model_params}

        # create output directory
        self.result_dir = join(constants.OUTPUT_ROOT, self.runtime_params.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        self.save_params()
        self.model_params.pop("type")

        # load data
        self.features, self.labels, self.participant_id = self.get_dataset()
        self.cr_splits = self.get_cross_validation_splits()
        self.input_shape = self.features[0].shape
        self.sign_to_prediction_id = get_sign_encoder()
        self.num_labels = len(self.sign_to_prediction_id)

        # define model runtime_params
        self.loss_function = "sparse_categorical_crossentropy"
        self.metrics = ["acc", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="t3_acc")]

        # define training runtime_params
        self.default_callbacks = self.get_default_callbacks()

    def save_params(self) -> None:
        # save runtime_params
        pprint(self.all_params)
        with open(join(self.result_dir, "all_params.json"), "w") as param_fp:
            json.dump(self.all_params, param_fp, indent=2)

    def get_dataset(self) -> Tuple[np.ndarray, ...]:
        """Load features and labels from a numpy file."""
        dataset = np.load(self.runtime_params.train_data, allow_pickle=True).item()
        features = np.array(dataset["features"], dtype=np.float32)
        assert not np.any(np.isnan(features)), "dataset contains NAN"
        labels = np.array(dataset["labels"], dtype=np.uint8)
        participant_id = np.array(dataset["participant_id"])
        with open(join(self.result_dir, "data.conf"), "w") as fd:
            json.dump(dataset["config"], fd, indent=2)
        return features, labels, participant_id

    def get_cross_validation_splits(self) -> Dict[int, Dict[str, list]]:
        sgkf = StratifiedGroupKFold(n_splits=self.runtime_params.cross_validation_splits, shuffle=True)
        data_idxs = range(len(self.features))
        cr_splits = {
            i: {"train": train_idxs, "val": val_idxs}
            for i, (train_idxs, val_idxs) in enumerate(sgkf.split(data_idxs, self.labels, self.participant_id))
        }
        return cr_splits

    def get_model(self) -> Model:
        if self.model_architecture == "fully_connected_v1":
            return FullyConnectedV1(**self.model_params)
        else:
            raise TypeError("Invalid Model type")

    def get_default_callbacks(self) -> list:
        """returns callbacks for training"""
        return [
            keras.callbacks.EarlyStopping(
                patience=self.runtime_params.early_stopping_patience, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=self.runtime_params.lr_reduce_patience, factor=self.runtime_params.lr_reduce_mult, verbose=1
            ),
        ]

    def save_model(self, model: Model, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_dir)

    def train(self) -> None:
        for fold_num, fold_idxs in self.cr_splits.items():
            logger.info(f"\n\nTraining for CV FOLD: {fold_num}\n")

            train_idxs, val_idxs = fold_idxs["train"], fold_idxs["val"]
            train_x, val_x = self.features[train_idxs], self.features[val_idxs]
            train_y, val_y = self.labels[train_idxs], self.labels[val_idxs]

            model = self.get_model()
            optimizer = keras.optimizers.Adam(self.runtime_params.lr)
            model.compile(optimizer, self.loss_function, metrics=self.metrics)

            callbacks = self.default_callbacks + [
                keras.callbacks.CSVLogger(join(self.result_dir, f"training_log_fold{fold_num:>02}.csv"))
            ]

            model.build((None, *self.input_shape))
            if fold_num == 0:
                model.summary()

            model.fit(
                train_x,
                train_y,
                validation_data=(val_x, val_y),
                epochs=self.runtime_params.epoch,
                callbacks=callbacks,
                batch_size=self.runtime_params.batch_size,
            )

            # save model
            save_dir = join(
                self.result_dir,
                f"fold{fold_num:>02}_{model.evaluate(val_x, val_y, verbose=0)[1]:.4f}".replace("0.", ""),
            )
            logger.info(f"Saving CV {fold_num} model to folder {save_dir}")
            self.save_model(model, save_dir)

            # Cleanup
            del model, train_x, train_y, val_x, val_y
            gc.collect()


if __name__ == "__main__":
    seed_it_all()
    Trainer().train()
