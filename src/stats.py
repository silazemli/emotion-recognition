import numpy as np
from preprocessing import preprocess
from split import get_ravdess_split
from pathlib import Path

stats_dir = Path("src/stats")
stats_dir.mkdir(exist_ok=True)

mels = []

train_files, _ = get_ravdess_split()

for file in train_files:
    mel = preprocess(file)
    mels.append(mel)

mels = np.stack(mels)

mean = mels.mean(axis=(0, 2))
std = mels.std(axis=(0, 2))

np.save(stats_dir / "mel_mean.npy", mean)
np.save(stats_dir / "mel_std.npy", std)