#! /usr/bin/env python

import maple
import numpy as np
import pandas as pd
import sounddevice as sd

from pathlib import Path

from scipy.io.wavfile import write as wav_write
from scipy.io.wavfile import read as wav_read


class OwnerRecordings(object):
    def __init__(self):
        self.dir = maple.owner_recordings_dir
        self.dir.mkdir(exist_ok=True)

        self.load()


    def write(self, name, data, fs, sentiment=None):
        """Write a numpy array to a .wav file

        Parameters
        ==========
        name : str
            Just the basename of the file, with no extensions

        data : numpy array
            A numpy array

        fs : int
            sampling freq

        sentiment : str, None
            label for the owner recording
        """

        output = self.dir/(name+'.wav')
        wav_write(output, fs, data)

        if self.summary is not None:
            new_summary = self.summary.append({'name': name, 'sentiment': sentiment}, ignore_index=True)
            new_summary.to_csv(self.dir/'summary.txt', sep='\t', index=False)
        else:
            pd.DataFrame({'name': [name], 'sentiment': [sentiment]}).to_csv(self.dir/'summary.txt', sep='\t', index=False)

        self.load()


    def load(self):
        self.recs = list(self.dir.glob('*.wav'))
        self.num = len(self.recs)
        self.names = [x.stem for x in self.recs]

        self.arrays = {}
        for rec in self.recs:
            _, data = wav_read(rec)
            self.arrays[rec.stem] = data

        if (self.dir/'summary.txt').exists():
            self.summary = pd.read_csv(self.dir/'summary.txt', sep='\t')
            self.sentiment = dict(zip(self.summary['name'], self.summary['sentiment']))
        else:
            self.summary = None
            self.sentiment = None


    def play(self, name, blocking=False):
        sd.play(self.arrays[name], blocking=blocking)


    def play_random(self, blocking=False, sentiment=None):
        if sentiment is None:
            names = self.names
        else:
            names = [k for k, v in self.sentiment.items() if v == sentiment]

        if not len(names):
            return

        choice = np.random.choice(names)
        self.play(choice, blocking=blocking)

        return choice


