import datetime
import numpy as np

import dgl
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class EarlyStopping:
    def __init__(self, prefix, patience=10):
        dt = datetime.datetime.now()
        Path("callbacks").mkdir(parents=True, exist_ok=True)
        filename = "callbacks/{}_early_stopping_{}_{:02d}-{:02d}-{:02d}.pth".format(
            prefix, datetime.datetime.now().date(), dt.hour, dt.minute, dt.second
        )

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def step(self, score, model):
        self.timestep += 1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(
            {"model_state_dict": model.state_dict(), "timestep": self.timestep},
            self.filename,
        )

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename)["model_state_dict"])


def get_stratified_folds(y, bins=10):
    return np.searchsorted(np.percentile(y, np.linspace(100 / bins, 100, bins)), y)


def get_samples(graphs, labels):
    return [(graph, label) for graph, label in zip(graphs, labels)]


def collate_fn(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
