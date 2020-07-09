#! /usr/bin/env python

import sounddevice as sd

from pathlib import Path

CHUNK = 2**11
RATE = 44100
sd.default.samplerate = RATE

owner_recordings_dir = Path(__file__).parent.parent / 'owner_recordings'
db_dir = Path(__file__).parent.parent / 'data' / 'dbs'

db_structure = {
    'self': {
        'names': ['id'],
        'types': ['text'],
    },
    'events': {
        #'names': ['t_start', 't_end', 't_len', 'energy', 'power_std', 'power_mean', 'class', 'audio'],
        #'types': ['text', 'text', 'numeric', 'numeric', 'numeric', 'numeric', 'text', 'blob'],
        'names': ['t_start', 't_end', 't_len'],
        'types': ['text', 'text', 'numeric'],
    },
}
