import torch
from torch.utils.data import Dataset

from preprocessing import (
    load_audio,
    fix_length, make_mel
)

class RAVDESSDataset(Dataset):
    def __init__(self, files):
        self.files = files
        
        self.labels = [
            int(file.stem.split("-")[2]) - 1
            for file in files
        ]
    
        self.X = []
        self.y = []

        for file, label in zip(files, self.labels):
            audio = load_audio(file)
            audio = fix_length(audio)

            mel = make_mel(audio)

            self.X.append(torch.from_numpy(mel).unsqueeze(0))
            self.y.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]