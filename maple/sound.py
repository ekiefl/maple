#! /usr/bin/env python

import maple
import numpy as np
import pyaudio
import argparse


class LiveStream(object):
    def __init__(self, args = argparse.Namespace()):
        self.args = args

        self.p = pyaudio.PyAudio()
        self.stream = None


    def start(self):
        while True:
            data = np.fromstring(self.stream.read(maple.CHUNK),dtype=np.int16)
            self.process_data(data)


    def __enter__(self):
        self.stream = self.p.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = maple.RATE,
            input = True,
            frames_per_buffer = maple.CHUNK,
        )

        return self


    def __exit__(self, exception_type, exception_value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


    def process_data(self, data):
        peak=np.average(np.abs(data))*2
        bars="-"*int(2000*peak/2**16)
        print("%05d %s"%(peak,bars))

        x = data

        w = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x))

        max_freq = abs(freqs[np.argmax(w)] * maple.RATE)
        peak = max_freq
        bars="o"*int(6000*peak/2**16)
        print("%05d %s"%(peak,bars))


