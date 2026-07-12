import torch
from torch.utils.data import Dataset

from preprocessing import (
    load_audio, normalize,
    fix_length, make_mel
)
from augment import random_gain, add_noise

class RAVDESSDataset(Dataset):
    def __init__(
            self, files,
            mean, std,
            lazy=False,
            augment=False
    ):
        self.files = files
        self.mean = mean
        self.std = std
        self.lazy = lazy
        self.augment = augment
        
        self.labels = [
            int(file.stem.split("-")[2]) - 1
            for file in files
        ]

        if not self.lazy:
            self.X = []
            self.y = []

            for file, label in zip(files, self.labels):
                audio = load_audio(file)
                audio = fix_length(audio)

                mel = make_mel(audio)
                # mel = normalize(mel, mean, std)

                self.X.append(torch.from_numpy(mel).unsqueeze(0))
                self.y.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.lazy:
            audio = load_audio(self.files[idx])
            audio = fix_length(audio)

            if self.augment:
                audio = random_gain(audio)
                audio = add_noise(audio)

            mel = make_mel(audio)
            mel = normalize(mel, self.mean, self.std)

            return (
                torch.from_numpy(mel).unsqueeze(0),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )
        else:
            return self.X[idx], self.y[idx]