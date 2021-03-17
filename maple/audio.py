#! /usr/bin/env python

import maple
import sounddevice as sd

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def PSD(data, fs=None):
    if not fs: fs = maple.RATE
    return signal.welch(data, maple.RATE, scaling='density')


def plot_PSD(data, fs=None):
    if not fs: fs = maple.RATE
    f, Pwelch_spec = PSD(data, fs)

    plt.semilogy(f, Pwelch_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()


def denoise(data, bg_data):
    fft_data = np.fft.rfft(data)
    fft_bg_data = np.fft.rfft(bg_data, n=len(data))

    return np.fft.irfft(fft_data - fft_bg_data).astype(data.dtype)


def bandpass(data, lowcut, highcut):
    sos = signal.butter(10, [lowcut, highcut], 'bandpass', fs=maple.RATE, output='sos')
    return signal.sosfilt(sos, data).astype(data.dtype)


def get_spectrogram(audio, fs=None, log=False, flatten=True):
    if fs is None:
        fs = maple.RATE

    f, t, Sxx = signal.spectrogram(audio, fs)

    output = Sxx
    if log:
        output = np.log2(output)
    if flatten:
        output = output.flatten()

    return output














