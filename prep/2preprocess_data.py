import os
import pickle
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import xarray as xr
from matplotlib import path
from netCDF4 import Dataset
from scipy.ndimage import interpolation
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
from omegaconf import OmegaConf


cfg = OmegaConf.load('conf/config.yml')
src_dir = cfg.catalogue.raw
out_dir = cfg.catalogue.clean

sst_offset = 273.15
files = list(Path(src_dir).glob("**/*.nc"))

def crop_peninsula(arr):
    assert arr.shape == (3600, 7200)
    lat_from = 28
    lat_to = 45
    lon_from = 119
    lon_to = 138
    scale = 20  # OSTIA data's transformation scale: 3600/20, 7200/20 -> 180, 360
    return arr[
        1800 + lat_from * scale : 1800 + lat_to * scale,
        3600 + lon_from * scale : 3600 + lon_to * scale,
    ]

def validate_files(files):
    for file in files:
        year = int(file.parent.stem)
        dt = file.stem.split("-")[0]
        assert year == int(dt[:4])
        assert len(dt) in [8, 14]  # yyyymmdd or yyyymmddHHMMSS
        if len(dt) == 14:
            assert dt[-6:] == "120000"
    return 1

assert validate_files(files)

with tqdm.tqdm(iterable=files) as pbar:
    for i, file in enumerate(files):
        with xr.open_dataset(file) as ds:
            # ds = xr.open_dataset('../input/2019/20191231-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc')
            start_time = ds.start_time
            if re.match("\d+T\d+Z", start_time):
                dt = pd.Timestamp(ds.start_time).replace(tzinfo=None)
            else:
                yymmdd = re.findall("\d{4}-\d{2}-\d{2}", ds.start_date)[0]
                hh24miss = re.findall("\d{2}:\d{2}:\d{2}", ds.start_time)[0]
                dt = pd.Timestamp("{} {}".format(yymmdd, hh24miss))
            sst = ds.variables["analysed_sst"].values[0]
            sea_mask = (ds.mask.values == 1)[0]  # 1 eq sea
            sst = crop_peninsula(sst)
            sea_mask = crop_peninsula(sea_mask)
            dt_str = dt.strftime("%Y%m%d%H%M%S")
            dest = f"../output/{dt_str}.npz"
            np.savez(dest, sst=sst, mask=sea_mask)
            pbar.update(1)
