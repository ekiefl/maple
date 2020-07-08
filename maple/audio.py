#! /usr/bin/env python

import maple
import sounddevice as sd

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs=None, order=5):
    if not fs: fs = maple.RATE
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs=None, order=5):
    if not fs: fs = maple.RATE
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def PSD(data, fs=None):
    if not fs: fs = maple.RATE
    return signal.welch(data, maple.RATE, scaling='spectrum')


def plot_PSD(data, fs=None):
    if not fs: fs = maple.RATE
    f, Pwelch_spec = PSD(data, fs)

    plt.semilogy(f, Pwelch_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

