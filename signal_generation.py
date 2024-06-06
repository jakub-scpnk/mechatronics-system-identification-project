# -*- coding: utf-8 -*-
"""
Created on Thu May 1 19:34:06 2024

@author: Jakub

This module contains functions for generating signals for testing SCD and
Coherence functions.

It's part of project for Mechatronic Systems Identification classes on AGH UST.

"""

import numpy as np
import matplotlib.pyplot as plt


def generate_cyclostationary_signal(freq, amp, mod_freq, fs, duration):
    """
    Generates a cyclostationary signal.

    Parameters
    ----------
    freq : list
        list of frequencies of the sinusoids.
    amp : list
        list of amplitudes of the sinusoids.
    mod_freq : float
        frequency of modulation signal (cosine wave).
    fs : int
        sampling rate.
    duration : float
        duration of signal in seconds.

    Returns
    -------
    signal : np.ndarray
        cyclostationary signal.
    t : np.ndarray
        time vector.

    """
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t)

    for f, a in zip(freq, amp):
        sinusoid = a * np.sin(2 * np.pi * f * t)
        mod_sinusoid = sinusoid * (1 + 0.5 * np.cos(2 * np.pi * mod_freq * t))
        signal += mod_sinusoid

    return signal, t

def plot_cyclostationary_signal(tv, signal):
    """
    Plots a cyclostationary signal.

    Parameters
    ----------
    tv : np.ndarray
        time vector.
    signal : np.ndarray
        signal.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10, 4))
    plt.plot(tv, signal)
    plt.title('Input Cyclostationary Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [-]')
    plt.grid(True)
    plt.show()

def generate_ergodic_signals(fs=1000, duration=10, seed=0):
    """
    Generates two signals: an input signal and an output signal, which is a
    nonlinear transformation of the input.

    Parameters
    ----------
    fs : int
        sampling frequency. The default is 1000.
    duration : float
        duration of signal in seconds. The default is 10.
    seed : float
        seed for random number generator. The default is 0.

    Returns
    -------
    t : np.ndarray
        time vector.
    input_signal : np.ndarray
        generated imput signal.
    output_signal : np.ndarray
        nonlinear transition of the imput.

    """
    # Time vector
    t = np.linspace(0, duration, fs * duration)
    np.random.seed(seed)  # For reproducibility

    # Generate input signal
    sin_component_1 = np.sin(2 * np.pi * 5 * t)
    sin_component_2 = 0.5 * np.sin(2 * np.pi * 20 * t)
    noise_component = 0.3 * np.random.normal(size=len(t))
    
    input_signal = sin_component_1 + sin_component_2 + noise_component

    # Define a nonlinear transfer function
    def nonlinear_system(x):
        return np.tanh(x) + 0.1 * np.sin(5 * x)

    # Generate output signal
    output_signal = nonlinear_system(input_signal)

    return t, input_signal, output_signal

def plot_ergodic_signals(t, input_signal, output_signal):
    """
    Plots the input and output signals.

    Parameters
    ----------
    t : np.ndarray
        time vector.
    input_signal : np.ndarray
        generated imput signal.
    output_signal : np.ndarray
        nonlinear transition of the imput.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, input_signal, label='Input Signal')
    plt.title('Input Signal')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, output_signal, label='Output Signal', color='orange')
    plt.title('Output Signal')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_bpsk_signal(bitstream, bit_duration, fs):
    """
    Generates a BPSK signal from a given bitstream.

    Parameters
    ----------
    bitstream : list or np.ndarray
        List or array of binary bits (0s and 1s).
    bit_duration : float
        Duration of each bit in seconds.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    signal : np.ndarray
        The generated BPSK signal.
    t : np.ndarray
        The time vector corresponding to the signal.
    """
    # Time vector for each bit
    t_bit = np.arange(0, bit_duration, 1/fs)
    
    # Create a phase for each bit: 0 for bit 1, pi for bit 0
    phase = np.array([0 if bit == 1 else np.pi for bit in bitstream])
    
    # Generate BPSK signal
    signal = np.hstack([np.cos(2 * np.pi * t_bit + p) for p in phase])
    
    # Generate time vector for the entire signal
    t = np.arange(0, bit_duration * len(bitstream), 1/fs)
    
    return signal, t