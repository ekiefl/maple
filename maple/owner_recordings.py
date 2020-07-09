#! /usr/bin/env python

import maple

from pathlib import Path

from scipy.io.wavfile import write as wav_write
from scipy.io.wavfile import read as wav_read


class OwnerRecordings(object):
    def __init__(self):
        self.dir = maple.owner_recordings_dir
        self.dir.mkdir(exist_ok=True)

        self.recs = list(self.dir.glob('*.wav'))
        self.num = len(self.recs)

        self.arrays = {}


    def write(self, name, data, fs):
        """Write a numpy array to a .wav file

        Parameters
        ==========
        name : str
            Just the basename of the file, with no extensions

        data : numpy array
            A numpy array

        fs : int
            sampling freq
        """

        output = self.dir/(name+'.wav')
        wav_write(output, fs, data)

        self.num += 1
        self.recs.append(output)


    def load(self):
        for rec in self.recs:
            _, data = wav_read(rec)
            self.arrays[rec.stem] = data

