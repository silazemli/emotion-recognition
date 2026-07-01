import librosa
import numpy as np

SR = 16000
DURATION = 4.0

def load_audio(path, sr):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def fix_length(audio, target_len):
    if len(audio) > target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')

def make_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db

def normalize(mel, mean, std):
    return (mel - mean) / (std + 1e-9)

def preprocess(path, sr=SR, duration=DURATION, mean=None, std=None):
    target_len = int(sr * duration)
    
    audio = load_audio(path, sr)
    audio = fix_length(audio, target_len)
    
    mel = make_mel(audio, sr)
    # if mean is not None:
    #     mel = normalize(mel, mean, std)

    return mel
