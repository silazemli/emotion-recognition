import librosa
import numpy as np

SR = 16000
DURATION = 4.0

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def fix_length(audio, sr=SR, duration=DURATION):
    target_len = int(sr*duration)
    
    if len(audio) > target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')

def make_mel(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=512,
        hop_length=160
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    
    return mel.astype(np.float32)