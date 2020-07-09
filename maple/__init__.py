#! /usr/bin/env python

import sounddevice as sd

from pathlib import Path

CHUNK = 2**11
RATE = 44100
sd.default.samplerate = RATE

owner_recordings_dir = Path(__file__).parent.parent / 'owner_recordings'
