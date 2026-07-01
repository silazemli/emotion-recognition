from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import RAVDESSDataset
from model import CNN


data_dir = Path("data/audio_speech_actors_01-24")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def add_noise(x, sigma=0.012):
    return x + sigma * torch.randn_like(x)

wav_files = list(data_dir.rglob("*.wav"))

train_size = int(0.8 * len(wav_files))
test_size = len(wav_files) - train_size

train_files, test_files = random_split(
    wav_files,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

mean, std = RAVDESSDataset.compute_mean_std(train_files)

train_dataset = RAVDESSDataset(
    train_files,
    emotion_map,
    mean,
    std
)

test_dataset = RAVDESSDataset(
    test_files,
    emotion_map,
    mean,
    std
)

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

for epoch in range(13):
    model.train()
    total_loss = 0

    correct = 0
    total = 0

    for x, y in train_loader:
        x = add_noise(x, sigma=0.01)

        x = x.to(device)
        y = y.to(device)

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
