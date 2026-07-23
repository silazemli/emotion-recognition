# Speech Emotion Recognition using CNNs

## Overview

This project implements a convolutional neural network (CNN) for speech emotion recognition using the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Speech recordings are converted into normalized Mel spectrograms, which are then used to train a CNN to classify one of eight emotion categories.

## Features
 - Fixed-length audio preprocessing with Librosa
 - Log Mel spectrogram generation and normalization
 - CNN classifier implemented in PyTorch
 - SpecAugment data augmentation
 - Speaker-independent evaluation
 - Learning rate scheduling
 - Early stopping
 - Automatic model checkpointing

## Dataset 

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) emotional speech dataset.

To evaluate generalization to unseen speakers, the dataset is split by actor rather than by individual recordings:

- **Training:** Actors 1–20
- **Testing:** Actors 21–24

This speaker-independent split prevents the model from learning speaker-specific characteristics and provides a more realistic evaluation of performance.

Additional exploratory data analysis is available in `eda.ipynb`.

## Preprocessing

Each audio sample undergoes the following preprocessing steps:

1. Load audio as mono.
2. Resample to 16 kHz.
3. Trim or zero-pad each clip to 4 seconds.
4. Generate a 128-bin Mel spectrogram.
5. Convert the spectrogram to the decibel scale.
6. Normalize the spectrogram to zero mean and unit variance.

The resulting normalized Mel spectrogram is used as input to the CNN.

![Example Mel Spectrogram](images/mel_spectrogram.png)

## Model Architecture

The model is a convolutional neural network consisting of four convolutional blocks followed by two fully connected layers.

```text
Input Mel Spectrogram
        │
Conv2D → BatchNorm → ReLU → MaxPool
        │
Conv2D → BatchNorm → ReLU → MaxPool
        │
Dropout
        │
Conv2D → BatchNorm → ReLU → MaxPool
        │
Conv2D → BatchNorm → ReLU → MaxPool
        │
Adaptive Average Pooling
        │
Fully Connected (128)
        │
Dropout
        │
Fully Connected (8)
```

## Training
The model is trained using the following configuration:

| Parameter | Value |
|-----------|------:|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Batch size | 32 |
| Loss function | CrossEntropyLoss |
| Learning rate scheduler | ReduceLROnPlateau |
| Early stopping | Patience = 25 epochs |

During training, SpecAugment is applied by randomly masking time and frequency regions of the spectrogram to improve generalization.

## Results

The final model achieved a test accuracy of 56.7% on a speaker-independent evaluation split, where recordings from actors 1–20 were used for training and actors 21–24 were reserved for testing.

Performance varied across emotion classes. The model achieved the highest recall for *Surprised*, followed by *Calm*, while *Neutral* and *Happy* proved to be the most challenging emotions to classify. The model performed best on emotions with more distinctive acoustic patterns, while emotions with similar vocal expressions were more frequently confused. The confusion matrix shows that many misclassifications occur between related emotions, particularly *Calm* and *Sad*.

A confusion matrix illustrating the model's predictions is shown below.

![Confusion Matrix](images/confusion_matrix_normalized.png)

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```