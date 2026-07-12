import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RAVDESSDataset
from model import CNN
from split import get_ravdess_split
from augment import spec_augment

import numpy as np
import time

LAZY_DATASET = False
WAVEFORM_AUGMENT = False
SPEC_AUGMENT = True

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
torch.xpu.manual_seed(seed)

train_files, test_files = get_ravdess_split()

mean = np.load("src/stats/mel_mean.npy")
std = np.load("src/stats/mel_std.npy")

train_dataset = RAVDESSDataset(
    train_files,
    mean,
    std,
    lazy=LAZY_DATASET,
    augment=WAVEFORM_AUGMENT
)

test_dataset = RAVDESSDataset(
    test_files,
    mean,
    std,
    lazy=LAZY_DATASET,
    augment=False
)

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

device = "xpu"

model = CNN(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

best_acc = 0.0
patience = 15
epochs_without_improvement = 0

for epoch in range(200):
    start = time.time()
    model.train()
    total_loss = 0

    correct = 0
    total = 0

    freq_mask_width = 10
    time_mask_width = 12

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if SPEC_AUGMENT:
            x = spec_augment(
                x,
                freq_max_width=freq_mask_width,
                time_max_width=time_mask_width
            )
        
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

    if test_acc > best_acc:
        best_acc = test_acc
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "models/best_model.pt")
        
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break

    scheduler.step(test_acc)

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"{epoch+1:3d} | "
        f"loss: {(total_loss / len(train_loader)):.2f} | "
        f"train / test: {train_acc:.3f} / {test_acc:.3f} | "
        f"et: {(time.time() - start):.0f}"
        f"{" | new best!" if epochs_without_improvement == 0 else ""}"
    )