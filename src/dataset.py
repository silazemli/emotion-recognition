import torch
from torch.utils.data import Dataset
from preprocessing import preprocess
import numpy as np

class RAVDESSDataset(Dataset):
    def __init__(self, files):
        self.X = []
        self.y = []

        for file in files:
            mel = preprocess(file)

            self.X.append(torch.tensor(mel).unsqueeze(0))
            self.y.append(int(file.stem.split("-")[2]) - 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]