#! /usr/bin/env python

import maple
import sounddevice as sd

import numpy as np
import noisereduce.noisereducev1 as nr
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

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
    return nr.reduce_noise(
        audio_clip=data.astype(float),
        noise_clip=bg_data.astype(float),
        pad_clipping=True,
    ).astype(maple.ARRAY_DTYPE)


def bandpass(data, lowcut, highcut):
    sos = signal.butter(10, [lowcut, highcut], 'bandpass', fs=maple.RATE, output='sos')
    return signal.sosfilt(sos, data).astype(data.dtype)


def get_spectrogram(audio, fs=None, log=False, flatten=False):
    if fs is None:
        fs = maple.RATE

    f, t, Sxx = signal.spectrogram(audio, fs)

    output = Sxx
    if log:
        output = np.log2(output)
    if flatten:
        output = output.flatten()

    return f, t, output


def get_fourier(audio, fs=None):
    """Return the amplitude of fourier transformed data, along with frequencies

    Returns
    =======
    out : amplitudes, frequencies
    """

    N = len(audio)

    fs = fs if fs is not None else maple.RATE
    T = 1/fs

    faudio = fft(audio)[:N//2]
    amps = 2/N * np.abs(faudio)
    f = fftfreq(N, T)[:N//2]

    return amps, f












