import numpy as np

def time_mask(spec, max_width):
    if max_width == 0:
        return spec
    
    time = spec.shape[3]

    t = np.random.randint(0, max_width)
    if t == 0:
        return spec
    t0 = np.random.randint(0, time - t)

    spec[:, :, :, t0:t0+t] = 0
    return spec

def freq_mask(spec, max_width):
    if max_width == 0:
        return spec

    freq = spec.shape[2]

    f = np.random.randint(0, max_width)
    if f == 0:
        return spec
    f0 = np.random.randint(0, freq - f)

    spec[:, :, f0:f0+f, :] = 0
    return spec

def spec_augment(spec, freq_max_width=7, time_max_width=11):
    spec = spec.clone()
    spec = time_mask(spec, time_max_width)
    spec = freq_mask(spec, freq_max_width)
    return spec

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
