import argparse
import datetime
import os
import random
import sys
import uuid
from itertools import product
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf, DictConfig
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import load_model
import hydra


def generate_stem(cfg):
    params = {
        k: f"{v:.0E}" if isinstance(v, float) else v
        for k, v in cfg.items()
        if k != "mask"
    }
    return "&".join([f"{k}={v}" for k, v in params.items()])


def generate_filedataset():
    source = "scratch/input/clstm_X_tr_te.npz"
    out_dir = "scratch/input"
    arr = np.load(source)
    train = np.concatenate([arr["x_train"], arr["y_train"]], axis=1)
    test = np.concatenate([arr["x_test"], arr["y_test"]], axis=1)

    train_size = int(train.shape[0] * 0.8)
    val = train[train_size:]
    train = train[:train_size]

    names = ["train", "val", "test"]
    c = 0
    for name, data in zip(names, [train, val, test]):
        for img in data:
            dest = Path(out_dir) / name / f"{c}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(img, dest)
            c += 1


def dset_generator(pred_hour, mode="train"):
    lat, lon = 75, 104

    def inner():
        source = {}
        source["train"] = "scratch/input/train"
        source["val"] = "scratch/input/val"
        source["test"] = "scratch/input/test"
        files = [x for x in Path(source[mode]).iterdir()]
        for file in files:
            pass
            x = joblib.load(file)

            label_idx = {
                6: -1,
                3: -2,
                1: -3,
            }[pred_hour]
            y = x[label_idx]
            x = x[:4]

            x = x.reshape(x.shape[0], lat, lon)[..., np.newaxis]
            y = y.reshape(lat, lon)[..., np.newaxis]
            yield x, y

    return inner


@hydra.main("conf", "config")
def end_to_end(cfg: DictConfig):
    pred_hour = cfg.model.pred_hour
    train_gen = dset_generator(pred_hour=1, mode="train")
    val_gen = dset_generator(pred_hour=1, mode="val")
    test_gen = dset_generator(pred_hour=1, mode="test")

    train_set = tf.data.Dataset.from_generator(
        train_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([4, 75, 104, 1]), tf.TensorShape([75, 104, 1])),
    )

    val_set = tf.data.Dataset.from_generator(
        val_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([4, 75, 104, 1]), tf.TensorShape([75, 104, 1])),
    )

    test_set = tf.data.Dataset.from_generator(
        test_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([4, 75, 104, 1]), tf.TensorShape([75, 104, 1])),
    )

    train_size = len([x for x in train_set])
    val_size = len([x for x in val_set])
    test_size = len([x for x in test_set])

    train_ds = (
        train_set.shuffle(train_size)
        .repeat()
        .batch(cfg.model.batch_size, drop_remainder=True)
    )

    val_ds = val_set.batch(cfg.model.batch_size, drop_remainder=True)
    test_ds = test_set.batch(cfg.model.batch_size, drop_remainder=True)

    input_shape = train_ds.element_spec[0].shape[1:]
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)
    # tfa.layers.InstanceNormalization(axis=3) when used with non-scaled input data
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        inputs = tf.keras.Input(input_shape)
        # x = inputs
        x = None
        for i, _ in enumerate(range(4)):
            x = tf.keras.layers.ConvLSTM2D(
                filters=cfg.model.filters,
                kernel_size=cfg.model.kernel_size,
                activation=cfg.model.activation,
                dilation_rate=cfg.model.dilation_rate,
                kernel_initializer="he_uniform"
                if cfg.model.activation == "relu"
                else "glorot_uniform",
                padding="same",
                return_sequences=True,
            )(inputs if i == 0 else x)
            if cfg.model.norm == "batch":
                # x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.experimental.SyncBatchNormalization()(x)
            elif cfg.model.norm == "group":
                groups = {15: 5, 16: 4, 17: 17, 18: 6, 19: 19, 20: 5}[cfg.model.filters]
                x = tfa.layers.GroupNormalization(groups=groups, axis=-1)(x)

        outputs = tf.keras.layers.ConvLSTM2D(
            filters=1,
            kernel_size=cfg.model.kernel_size,
            activation="linear",
            padding="same",  # sigmoid -> linear
            return_sequences=False,
            kernel_initializer="glorot_uniform",
        )(x)
        # skip = inputs[:,-1,...]
        # skip = tf.keras.layers.GaussianNoise(
        # stddev=1e-3)(skip)
        # outputs = x + skip
        # outputs = x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        loss_object = tf.keras.losses.MeanAbsoluteError(
            # reduction=tf.keras.losses.Reduction.NONE,
            tf.keras.losses.Reduction.SUM
        )

        model.compile(
            loss="mae", optimizer=tf.keras.optimizers.Adam(cfg.model.learning_rate)
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", min_delta=1e-3
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, mode="min", min_lr=1e-5
        ),
        # tf.keras.callbacks.ModelCheckpoint(filepath='../models/', monitor='val_loss', save_best_only=True),
    ]

    steps_per_epoch = train_size // cfg.model.batch_size
    validation_steps = val_size // cfg.model.batch_size

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=100,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    val_score = pd.DataFrame(history.history)["val_loss"].min()
    Path(cfg["save_dir"]).mkdir(exist_ok=True)

    payload = {**cfg}
    payload["val_loss"] = val_score
    payload["finished_dt"] = datetime.datetime.now()
    joblib.dump(payload, Path(cfg["save_dir"]) / "result")
    model_dest = os.path.join(cfg["save_dir"], "model.h5")
    model.save(model_dest)


if __name__ == "__main__":
    main()
