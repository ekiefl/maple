#! /usr/bin/env python

import sounddevice as sd

CHUNK = 2**11
RATE = 44100
sd.default.samplerate = RATE
