# -*- coding: utf-8 -*-
"""
Created on Thu May 2 00:42:31 2024

@author: Jakub
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence
from numpy.lib.stride_tricks import as_strided


def sliding_window(x, w, s):
    shape = ((x.shape[0] - w) // s + 1, w)
    strides = (x.strides[0] * s, x.strides[0])
    return as_strided(x, shape=shape, strides=strides)

def fam(x, Np, L, N=None):
    """
    Calculate the Spectral Correlation Density (SCD) using the FFT Accumulation
    Method (FAM).
    
    Code was drevied from
    https://github.com/avian2/spectrum-sensing-methods/blob/master/sensing/utils.py
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
    Np : int
        Length of the FFT window.
    L : int
        Step size between successive FFT windows.
    N : int, optional
        Length of the FFT. If not specified, it will be calculated based on L
        and P.

    Returns
    -------
    f : np.ndarray
        Frequency vector.
    alpha : np.ndarray
        Cyclic frequency vector.
    Sx : np.ndarray
        Spectral Correlation Density matrix.
    """
    # Input channelization
    xs = sliding_window(x, Np, L)
    
    if N is None:
        Pe = int(np.floor(np.log2(xs.shape[0])))
        P = 2**Pe
        N = L * P
    else:
        P = N // L
    
    P = min(P, xs.shape[0])  # Ensure P does not exceed number of segments
    
    xs2 = xs[:P, :]

    # Windowing
    w = np.hamming(Np)
    w /= np.sqrt(np.sum(w**2))
    xw = xs2 * np.tile(w, (P, 1))

    # First FFT
    XF1 = np.fft.fft(xw, axis=1)
    XF1 = np.fft.fftshift(XF1, axes=1)

    # Calculating complex demodulates
    f = np.arange(Np) / float(Np) - 0.5
    t = np.arange(P) * L
    f = np.tile(f, (P, 1))
    t = np.tile(t.reshape(P, 1), (1, Np))
    XD = XF1 * np.exp(-1j * 2 * np.pi * f * t)

    # Calculating conjugate products, second FFT and the final matrix
    Sx = np.zeros((Np, 2 * N), dtype=complex)
    Mp = N // Np // 2

    for k in range(Np):
        for l in range(Np):
            XF2 = np.fft.fft(XD[:, k] * np.conjugate(XD[:, l]))
            XF2 = np.fft.fftshift(XF2)
            XF2 /= P

            i = int((k + l) / 2)
            a = int(((k - l) / float(Np) + 1) * N)

            Sx[i, a - Mp:a + Mp] = XF2[P // 2 - Mp:P // 2 + Mp]

    f = np.fft.fftfreq(Np, d=1 / Np) - 0.5
    alpha = np.fft.fftfreq(2 * N, d=1 / (2 * N))

    return f, alpha, Sx

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

def plot_scd(signal, Np, L, N):
    """
    Calculate and plot the Spectral Correlation Density (SCD).

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    Np : int
        Length of the FFT window.
    L : int
        Step size between successive FFT windows.
    N : int
        Length of the FFT.
    """
    f, alpha, SCD = fam(signal, Np=Np, L=L, N=N)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(SCD), extent=[alpha[0], alpha[-1], f[0], f[-1]],
               aspect='auto',
               origin='lower')
    plt.title('Spectral Correlation Density')
    plt.xlabel('Normalized Cyclic Frequency')
    plt.ylabel('Normalized Frequency')
    plt.colorbar(label='Magnitude')
    plt.show()

def plot_coherence(signal1, signal2, fs, nperseg):
    """
    Calculate and plot coherence between two signals using both scipy and
    manual methods.

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
    """
    # Calculate coherence using scipy
    f_scipy, Cxy_scipy = calculate_coherence_scipy(signal1,
                                                   signal2,
                                                   fs,
                                                   nperseg)

    # Calculate coherence manually
    f_manual, Cxy_manual = calculate_coherence_manual(signal1,
                                                      signal2,
                                                      fs,
                                                      nperseg)

    # Plot coherence
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.semilogy(f_scipy, Cxy_scipy)
    plt.title('Coherence Using scipy.signal.coherence')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogy(f_manual, Cxy_manual)
    plt.title('Coherence Calculated Manually')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.grid(True)

    plt.tight_layout()
    plt.show()