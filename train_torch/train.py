import os
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
import hydra


def calc_padding(dhw, kernel_size, dilation):
    # d_in + 2 * p - d * (k-1) = d_out
    padding = []
    for _, k, d in zip(dhw, kernel_size, dilation):
        p = d * (k - 1) / 2
        padding.append(int(p))
        # padding =  (o-1-i+k+(k-1)*(d-1)) //2
        # return padding
    return padding

cfg = OmegaConf.load('conf/config.yml')
dim = cfg.model

dwh = (4, 340, 380)
dim["padding"] = calc_padding(dwh, dim["kernel_size"], dim["dilation"])
dim["padding"][0] = 0

# n, c, d, h ,w
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
arr = np.load("../input3/input_output_day_{}.npz".format(dim["output_day"]))
x = arr["x"]
y = arr["y"]
mask = arr["mask"]
idx = arr["idx"]
idx = pd.to_datetime(idx)

x = x[:, np.newaxis, ...]
y = y[:, np.newaxis, np.newaxis, ...]

train_mask = idx.year < 2018
val_mask = idx.year == 2018
test_mask = idx.year == 2019

x_train, y_train = x[train_mask], y[train_mask]
x_val, y_val = x[val_mask], y[val_mask]
x_test, y_test = x[test_mask], y[test_mask]


class SeafogDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

trainset = SeafogDataset(x_train, y_train)
valset = SeafogDataset(x_val, y_val)
train_loader = DataLoader(trainset, batch_size=dim["batch_size"], shuffle=True)
val_loader = DataLoader(trainset, batch_size=dim["batch_size"], shuffle=False)

class GaussianNoise(nn.Module):
    def __init__(self, stddev=1e-4):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(
                torch.randn(x.size()) * self.stddev
            ).to(x.device)
        return x

class Ostia(nn.Module):
    def __init__(self, dim):
        super(Ostia, self).__init__()
        self.noise = GaussianNoise()
        self.main = nn.ModuleList()
        for i in range(2):
            in_channels = 20 if i != 0 else 1
            out_channels = 20
            self.main.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=dim["kernel_size"],
                    stride=1,
                    padding=dim["padding"],
                    bias=False,
                    dilation=dim["dilation"],
                )
            )
            self.main.append(nn.GroupNorm(num_groups=4, num_channels=out_channels))
        self.out = nn.Conv3d(
            out_channels,
            1,
            kernel_size=dim["kernel_size"],
            stride=1,
            padding=dim["padding"],
            bias=False,
            dilation=dim["dilation"],
        )
        # self.noise = GaussianNoise(0.01)

    def forward(self, x):
        skip = x[:, :, [-1], ...].cuda()
        skip = self.noise(skip)
        for l in self.main:
            x = l(x)
        x = self.out(x)
        return x + skip

net = Ostia(dim)
net = nn.DataParallel(net)
net.cuda()
# criterion = nn.MSELoss()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=dim["lr"])
gamma = 0.95
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.95, patience=1
)

losses = [np.inf]
min_delta = 1e-5
run = 0
patience = 3
batch_span = 4
for epoch in range(cfg.epochs):
    net.train()
    with tqdm.tqdm(total=len(train_loader)) as pbar:
        for batch_count, (inputs, target) in enumerate(train_loader):
            if batch_count % batch_span == 0:
                optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, target.cuda())
            loss.backward()
            if batch_count > 0 and batch_count % batch_span == batch_span - 1:
                optimizer.step()
            pbar.update(1)
    net.eval()
    with tqdm.tqdm(total=len(val_loader)) as pbar:
        eval_loss = torch.scalar_tensor(0, dtype=torch.float32)
        # eval_loss = torch.autograd.Variable(eval_loss)
        for inputs, target in val_loader:
            output = net(inputs)
            loss = criterion(output, target.cuda())
            eval_loss += loss.to("cpu").detach()
            pbar.update(1)
        eval_loss = eval_loss / len(val_loader)
        print("epoch: {}, eval_loss: {:.4f}".format(epoch, eval_loss))
        if eval_loss > min(losses) - min_delta:
            run += 1
        else:
            run = 0
        losses.append(eval_loss)
        if np.greater(run, patience):
            break
    scheduler.step(eval_loss)
preds = []
net.eval()
with tqdm.tqdm(total=x_test.shape[0]) as pbar:
    for inputs in x_test:
        net.eval()
        intensor = torch.FloatTensor(inputs).unsqueeze(0)
        output = net(intensor)
        preds.append(output.detach().to("cpu").numpy())
        pbar.update(1)

test_size = y_test.shape[0]
pred = np.concatenate(preds).reshape(test_size, 340, 380)
obs = y_test.reshape(test_size, 340, 380)

pixel_max = arr["max"]
pixel_min = arr["min"]
pixel_span = pixel_max - pixel_min

obs = (obs + 0.5) * pixel_span + pixel_min
pred = (pred + 0.5) * pixel_span + pixel_min

delta = np.mean(np.abs(obs - pred), axis=0)
mean_delta = np.where(mask, delta, np.nan)
plt.close("all")
fig, ax = plt.subplots()
cmap = plt.cm.rainbow
cmap.set_bad(color="black")
pos = ax.imshow(mean_delta, cmap=cmap, vmin=0, vmax=1)
fig.colorbar(pos, ax=ax)

pred_flat = pred[:, mask].reshape(test_size, -1)
test_flat = obs[:, mask].reshape(test_size, -1)

metrics = {}
metrics["mae"] = mean_absolute_error(test_flat, pred_flat)
metrics["mse"] = mean_squared_error(test_flat, pred_flat)
metrics = pd.Series(metrics).to_frame().T
clean_dim = {
    k: v if not isinstance(v, float) else f"{v:.0E}" for k, v in dim.items()
}
stem = "&".join([f"{k}={v}" for k, v in dim.items()])
out = {}
# out['metrics'] = metric
out["metrics"] = metrics
joblib.dump(out, Path("../torch_product") / stem)
