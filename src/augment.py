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