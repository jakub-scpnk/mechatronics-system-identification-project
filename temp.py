import numpy as np
from scipy.signal import coherence
from numpy.lib.stride_tricks import as_strided

def calculate_coherence_scipy(signal1, signal2, fs, nperseg):
    """
    Calculate coherence between two signals using scipy.signal.coherence.

    Parameters
    ----------
    signal1 : np.ndarray
        First input signal.
    signal2 : np.ndarray
        Second input signal.
    fs : int
        Sampling frequency.
    nperseg : int
        Length of each segment for the FFT.

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies.
    Cxy : np.ndarray
        Coherence between signal1 and signal2.
    """
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    return f, Cxy


def calculate_coherence_manual(signal1, signal2, fs, nperseg):
    """
    Calculate coherence between two signals manually.

    Parameters
    ----------
    signal1 : np.ndarray
        First input signal.
    signal2 : np.ndarray
        Second input signal.
    fs : int
        Sampling frequency.
    nperseg : int
        Length of each segment for the FFT.

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies.
    Cxy : np.ndarray
        Coherence between signal1 and signal2.
    """
    from scipy.signal import get_window
    
    # Split signals into overlapping segments
    def segment_signal(signal, nperseg):
        step = nperseg // 2
        shape = ((signal.size - nperseg) // step + 1, nperseg)
        strides = (signal.strides[0] * step, signal.strides[0])
        return as_strided(signal, shape=shape, strides=strides)

    # Calculate the FFT of each segment
    def fft_segments(segments, nperseg):
        window = get_window('hann', nperseg)
        windowed_segments = segments * window
        return np.fft.rfft(windowed_segments, axis=-1)

    # Segment the signals
    segments1 = segment_signal(signal1, nperseg)
    segments2 = segment_signal(signal2, nperseg)

    # Compute FFT of each segment
    fft1 = fft_segments(segments1, nperseg)
    fft2 = fft_segments(segments2, nperseg)

    # Compute cross power spectral density
    Pxy = np.mean(fft1 * np.conj(fft2), axis=0)
    
    # Compute power spectral densities
    Pxx = np.mean(np.abs(fft1) ** 2, axis=0)
    Pyy = np.mean(np.abs(fft2) ** 2, axis=0)

    # Compute coherence
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)

    # Generate frequency vector
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    
    return f, Cxy


import matplotlib.pyplot as plt

# Example signals
fs = 1000  # Sampling frequency
t = np.linspace(0, 10, fs * 10)
signal1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(size=len(t))
signal2 = np.sin(2 * np.pi * 5 * t + np.pi / 4) + 0.5 * np.random.normal(size=len(t))

# Parameters for coherence calculation
nperseg = 256

# Calculate coherence using scipy
f_scipy, Cxy_scipy = calculate_coherence_scipy(signal1, signal2, fs, nperseg)

# Calculate coherence manually
f_manual, Cxy_manual = calculate_coherence_manual(signal1, signal2, fs, nperseg)

# Plot coherence
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogy(f_scipy, Cxy_scipy)
plt.title('Coherence Using scipy.signal.coherence')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Coherence')

plt.subplot(2, 1, 2)
plt.semilogy(f_manual, Cxy_manual)
plt.title('Coherence Calculated Manually')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Coherence')

plt.tight_layout()
plt.grid(True)
plt.show()