import argparse
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import xarray as xr
from matplotlib import font_manager as fm
from matplotlib import path
from matplotlib.pyplot import cm
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import norm
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import dilation
from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yml')

src_dir = cfg.catalogue.clean
out_dir = cfg.catalogue.model_in

sst_offset = 273.15
files = list(Path(src_dir).glob("**/*.npz"))

parser = argparse.ArgumentParser()
parser.add_argument("output_day", type=int)
if "ipykernel" in sys.argv[0]:
    output_day = 3
else:
    output_day = parser.parse_args().output_day

m = f"output day is {output_day}\n"
print(m * 5)

placeholder = 0

sst_list = []
indices = []
masks = []

for file in files:
    arr = np.load(file)
    sst = arr["sst"]
    sst -= sst_offset
    sst_list.append(sst[::-1, :])
    mask = arr["mask"]
    masks.append(mask[::-1, :])
    indices.append(pd.Timestamp(file.stem))

mask = np.stack(masks)
idx = np.array(indices)
summer_masks = [6 <= x.month <= 9 for x in idx]
summer_mask = mask[summer_masks].mean(axis=0).round()

# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(summer_mask)
mask = summer_mask[...]
# mask = binary_erosion(mask, iterations=5)
sst = np.stack(sst_list, axis=0)
sst[:, np.logical_not(mask)] = np.nan

coords = dict(time=idx, y=np.arange(340), x=np.arange(380))
# outlier qc
data = xr.DataArray(sst, dims=("time", "y", "x"), coords=coords)
data = data.sortby("time")
diff = np.diff(data, axis=0)
nan_mask = np.greater(np.abs(diff), 4)
nan_mask = np.pad(
    nan_mask, [(1, 0), (0, 0), (0, 0)], mode="constant", constant_values=[False]
)
keep_mask = np.logical_not(nan_mask)
data = data.where(keep_mask, np.nan)
data = data.interpolate_na(dim="time", method="linear")
full_idx = pd.date_range(idx.min(), idx.max(), freq="1d")
full_data = data.reindex(time=full_idx)

train_data = data.loc[data.time.dt.year < 2018]
pixel_min = train_data.min(dim="time")
pixel_max = train_data.max(dim="time")
pixel_span = pixel_max - pixel_min
scaled_data = (full_data - pixel_min) / pixel_span - 0.5
assert np.equal(np.isnan(scaled_data).mean(axis=(1, 2)), 1.0).sum()

# plt.imshow(full_data.mean(dim='time'))
# plt.imshow(scaled_data.mean(dim='time'))
assert np.isnan(scaled_data[626]).all()  # 626 범인

arr = scaled_data.values
tdim = 4 + 1 + output_day
# 4, 1, 1  tdim=6
shape = (arr.shape[0] - tdim + 1, tdim, *arr.shape[1:])
strides = (arr.strides[0], *arr.strides)
arr = as_strided(arr, shape, strides)
out_idx = full_idx[-arr.shape[0] :]

nan_idx = np.isnan(arr).all(axis=(2, 3))
skip_idx = nan_idx.any(axis=1)
assert skip_idx.sum() == tdim

arr = arr[~skip_idx]
arr[np.isnan(arr)] = placeholder
idx = out_idx[~skip_idx]
x = arr[:, :4, ...]
y = arr[:, -1, ...]
assert np.equal(x.shape[0], 3652 - tdim + 1 - tdim)  # 3641
np.savez(
    out_dir / "input_output_day_{}.npz".format(output_day),
    x=x,
    y=y,
    idx=idx,
    mask=mask,
    min=pixel_min,
    max=pixel_max,
    placeholder=placeholder,
)
