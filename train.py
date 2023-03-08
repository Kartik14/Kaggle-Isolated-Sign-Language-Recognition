import random
from os.path import join

import numpy as np
import tensorflow as tf

import constants
from model.fully_connected import build_classification_model


def train() -> None:
    # load data
    train_data = np.load(join(constants.DATA_ROOT, "feature_data_nonan.npy")).astype(np.float32)
    train_labels = np.load(join(constants.DATA_ROOT, "labels_data.npy")).astype(np.uint8)

    # train, val split
    total_samples = train_data.shape[0]
    validation_fraction = 0.1
    num_validation = int(total_samples * validation_fraction)
    num_train = total_samples - num_validation

    data_idxs = list(range(total_samples))
    random.shuffle(data_idxs)
    train_idxs, val_idxs = np.array(data_idxs[:num_train]), np.array(data_idxs[num_train:])

    val_x, val_y = train_data[val_idxs], train_labels[val_idxs]
    train_x, train_y = train_data[train_idxs], train_labels[train_idxs]

    # define mode
    model = build_classification_model()
    optimizer = tf.keras.optimizers.Adam(3.3e-4)
    loss_function = "sparse_categorical_crossentropy"
    model.compile(optimizer, loss_function, metrics="acc")

    # train
    callback_list = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, verbose=1),
    ]
    _ = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, callbacks=callback_list, batch_size=64)
    model.save("data/models/asl-signs")


if __name__ == "__main__":
    train()
