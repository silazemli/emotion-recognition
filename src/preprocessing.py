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

def add_noise(audio, min_snr=10, max_snr=25):
    noise = np.random.randn(len(audio))

    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)

    snr_db = np.random.uniform(min_snr, max_snr)

    scale = np.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))

    noisy_audio = audio + noise * scale

    return noisy_audio.astype(np.float32)

def random_gain(audio, min_db=-6, max_db=6):
    gain_db = np.random.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20)

    return audio * gain

def make_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=512,
        hop_length=160
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db

def preprocess(path, sr=SR, duration=DURATION):
    target_len = int(sr*duration)
    
    audio = load_audio(path, sr)
    audio = fix_length(audio, target_len)
    
    if np.random.random() < 0.5:
        audio = add_noise(audio)
    
    if np.random.random() < 0.0:
        audio = random_gain(audio)

    mel = make_mel(audio, sr)

    return mel
