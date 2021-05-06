#! /usr/bin/env python

import maple
import numpy as np
import pyaudio
import argparse

def get_mic_index():
    aaa = pyaudio.PyAudio()
    info = aaa.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if aaa.get_device_info_by_host_api_device_index(0, i).get('name') == "USB PnP Audio Device":
            return i
    else:
        raise Exception("USB PnP Audio Device not found")

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
            input_device_index = get_mic_index()
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

        #x = data

        #w = np.fft.fft(x)
        #freqs = np.fft.fftfreq(len(x))

        #max_freq = abs(freqs[np.argmax(w)] * maple.RATE)
        #peak = max_freq
        #bars="o"*int(6000*peak/2**16)
        #print("%05d %s"%(peak,bars))


