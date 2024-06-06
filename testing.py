# -*- coding: utf-8 -*-
"""
Created on Thu May 1 19:06:31 2024

@author: Jakub

This script tests scd and coherence functions.

It's part of project for Mechatronic Systems Identification classes on AGH UST.
"""

import analysis
import signal_generation as sg


# Parameters
frequencies = [50, 120]  # Frequencies of the sinusoids
amplitudes = [1, 0.5]    # Amplitudes of the sinusoids
MODULATION_FREQ = 5      # Frequency of the modulation signal
SAMPLE_RATE = 1000       # Sampling rate in Hz
DURATION = 1             # Duration in seconds

# Generate the cyclostationary signal
signal, tv = sg.generate_cyclostationary_signal(frequencies,
                                                amplitudes,
                                                MODULATION_FREQ,
                                                SAMPLE_RATE,
                                                DURATION)

# Generate the ergodic signals
t, input_signal, output_signal = sg.generate_ergodic_signals()

# Plot the input signals
sg.plot_cyclostationary_signal(tv, signal)
sg.plot_ergodic_signals(t, input_signal, output_signal)

# Example usage of SCD
analysis.plot_scd(signal, Np=256, L=128, N=1024)

# Example usage of Coherence
analysis.plot_coherence(input_signal,
                        output_signal,
                        SAMPLE_RATE,
                        nperseg=256)
