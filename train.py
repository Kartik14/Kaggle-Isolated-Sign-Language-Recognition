import gc
import json
import os
from os.path import join
from parser.train_parser import TrainParser
from pprint import pprint
from typing import Dict, Tuple

import keras
import numpy as np
from keras.models import Model
from sklearn.model_selection import StratifiedGroupKFold

import constants
from helper.logging import logger
from helper.utils import get_sign_encoder, seed_it_all
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
        self.features, self.labels, self.participant_id = self.get_dataset()
        self.cr_splits = self.get_cross_validation_splits()
        self.input_shape = self.features[0].shape
        self.sign_to_prediction_id = get_sign_encoder()
        self.num_labels = len(self.sign_to_prediction_id)

        # define model params
        self.model_architecture = self.params.model_architecture
        self.loss_function = "sparse_categorical_crossentropy"
        self.metrics = ["acc", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="t3_acc")]

        # define training params
        self.default_callbacks = self.get_default_callbacks()

    def get_dataset(self) -> Tuple[np.ndarray, ...]:
        """Load features and labels from a numpy file."""
        dataset = np.load(self.params.train_data, allow_pickle=True).item()
        features = np.array(dataset["features"], dtype=np.float32)
        assert not np.any(np.isnan(features)), "dataset contains NAN"
        features = self.normalise_data(features)
        labels = np.array(dataset["labels"], dtype=np.uint8)
        participant_id = np.array(dataset["participant_id"])
        return features, labels, participant_id

    def normalise_data(self, features: np.ndarray) -> np.ndarray:
        distribution_mean = np.mean(features, axis=0, keepdims=True)
        distribution_std = np.std(features, axis=0, keepdims=True)
        np.save(
            join(self.result_dir, "distribution_stats.npy"),
            {"distribution_mean": distribution_mean, "distribution_std": distribution_std},
        )
        logger.info(f'Saved distribution statistics to {join(self.result_dir, "distribution_stats.npy")}')
        return (features - distribution_mean) / distribution_std

    def get_cross_validation_splits(self) -> Dict[int, Dict[str, list]]:
        sgkf = StratifiedGroupKFold(n_splits=self.params.cross_validation_splits, shuffle=True)
        data_idxs = range(len(self.features))
        cr_splits = {
            i: {"train": train_idxs, "val": val_idxs}
            for i, (train_idxs, val_idxs) in enumerate(sgkf.split(data_idxs, self.labels, self.participant_id))
        }
        return cr_splits

    def get_model(self) -> Model:
        """defines and builds model layers"""
        if self.model_architecture == "fully_connected_v1":
            model_obj = FullyConnectedModel(input_shape=self.input_shape, n_labels=self.num_labels)
        else:
            raise RuntimeError
        model_obj.create_layers()
        return model_obj.build_model()

    def get_default_callbacks(self) -> list:
        """returns callbacks for training"""
        return [
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=1),
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
            optimizer = keras.optimizers.Adam(self.params.lr)
            model.compile(optimizer, self.loss_function, metrics=self.metrics)
            if fold_num == 0:
                model.summary()

            callbacks = self.default_callbacks + [
                keras.callbacks.CSVLogger(join(self.result_dir, f"training_log_fold{fold_num:>02}.csv"))
            ]
            model.fit(
                train_x,
                train_y,
                validation_data=(val_x, val_y),
                epochs=self.params.epoch,
                callbacks=callbacks,
                batch_size=self.params.batch_size,
            )

            # save model
            save_dir = join(
                self.result_dir,
                f"fold{fold_num:>02}_{model.evaluate(val_x, val_y, verbose=0)[1]:.4f}".replace("0.", ""),
            )
            logger.info(f"Saving CR {fold_num} model to folder {save_dir}")
            self.save_model(model, save_dir)

            # Cleanup
            del model, train_x, train_y, val_x, val_y
            gc.collect()


if __name__ == "__main__":
    seed_it_all()
    Trainer().train()
