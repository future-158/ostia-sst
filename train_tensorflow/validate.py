import argparse
import os
import random
import sys
from itertools import product
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

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

physical_devices = tf.config.list_physical_devices("GPU")
assert len(physical_devices) == 0


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


source_dir = Path("scratch/output2")
folders = [x for x in source_dir.iterdir() if x.is_dir()]

rows = []
for folder in folders:
    result = joblib.load(folder / "result")
    model_path = folder / "model.h5"

    row = {**result, "model_path": model_path}

    rows.append(row)

table = pd.DataFrame(rows).sort_values(["pred_hour", "val_loss"])

pred_hours = [3, 6]
for pred_hour in pred_hours:
    model_path = table.query("pred_hour==@pred_hour").iloc[0]["model_path"]
    test_gen = dset_generator(pred_hour=pred_hour, mode="test")
    test_set = tf.data.Dataset.from_generator(
        test_gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([4, 75, 104, 1]), tf.TensorShape([75, 104, 1])),
    )

    test_ds = test_set.batch(1024, drop_remainder=False)
    obs = tf.concat([x[1] for x in test_ds], axis=0)

    with tf.device("/cpu:0"):
        model = tf.keras.models.load_model(model_path)
        model.summary()
        pred = model.predict(test_ds)
        pred = tf.concat(pred, axis=0)

    pred = pred.numpy()
    scaler_file = "scratch/input/clstm_X_tr_te.npz"
    data_min = np.load(scaler_file)["min"]
    data_max = np.load(scaler_file)["max"]
    data_span = data_max - data_min

    pred = (pred * data_span) + data_min
    obs = (obs * data_span) + data_min

    diff = np.squeeze((obs - pred))
    abs_diff = np.abs(diff)
    mae = abs_diff.mean(axis=0)
