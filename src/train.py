from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import RAVDESSDataset
from model import CNN

import numpy as np

def time_mask(spec, max_width):
    if max_width == 0:
        return spec
    
    time = spec.shape[3]

    t = np.random.randint(0, max_width)
    t0 = np.random.randint(0, time - t)

    spec[:, :, :, t0:t0+t] = 0
    return spec

def freq_mask(spec, max_width):
    if max_width == 0:
        return spec

    freq = spec.shape[2]

    f = np.random.randint(0, max_width)
    f0 = np.random.randint(0, freq - f)

    spec[:, :, f0:f0+f, :] = 0
    return spec

def spec_augment(spec, freq_max_width=7, time_max_width=11):
    spec = time_mask(spec, time_max_width)
    spec = freq_mask(spec, freq_max_width)
    return spec

data_dir = Path("data/audio_speech_actors_01-24")

actor_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

train_actors = actor_dirs[:20]
test_actors  = actor_dirs[20:]

train_files = []
test_files = []

for actor in train_actors:
    train_files += list(actor.rglob("*.wav"))

for actor in test_actors:
    test_files += list(actor.rglob("*.wav"))

train_dataset = RAVDESSDataset(train_files)

test_dataset = RAVDESSDataset(test_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1.5e-4)

for epoch in range(20):
    model.train()
    total_loss = 0

    correct = 0
    total = 0

    if epoch < 5:
        freq_mask_width = 3
        time_mask_width = 5
    elif epoch < 8:
        freq_mask_width = 6
        time_mask_width = 10
    else:
        freq_mask_width = 8
        time_mask_width = 13

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        x = spec_augment(x, freq_max_width=freq_mask_width, time_max_width=time_mask_width)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    test_acc = correct / total

    print(
        f"Epoch {epoch+1:2d} | "
        f"Loss: {total_loss:.2f} | "
        f"Train: {train_acc:.3f} | "
        f"Test: {test_acc:.3f}"
    )

torch.save(model.state_dict(), "models/cnn_ravdess.pth")
