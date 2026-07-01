import torch
from torch.utils.data import Dataset
from preprocessing import preprocess
import numpy as np

class RAVDESSDataset(Dataset):
    def __init__(self, files, emotion_map, mean=None, std=None):
        self.X = []
        self.y = []

        for file in files:
            emotion = emotion_map[file.stem.split("-")[2]]

            mel = preprocess(file)

            self.X.append(torch.tensor(mel).unsqueeze(0))
            self.y.append(emotion)

        self.label2id = {e:i for i, e in enumerate(sorted(set(self.y)))}
        self.y = [self.label2id[e] for e in self.y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    @staticmethod
    def compute_mean_std(files):
        total = 0.0
        total_sq = 0.0
        count = 0

        for file in files:
            mel = preprocess(file)

            total += mel.sum()
            total_sq += np.square(mel).sum()
            count += mel.size
        
        mean = total / count
        std = np.sqrt(total_sq / count - mean**2)
        
        return mean, std
    