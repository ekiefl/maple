#! /usr/bin/env python

from pathlib import Path
import sounddevice as sd

CHUNK = 2**11
RATE = 44100
sd.default.samplerate = RATE

owner_recordings_dir = Path(__file__).parent.parent / 'owner_recordings'
