#! /usr/bin/env python

import numpy as np
import sounddevice as sd

from pathlib import Path

CHUNK = 2**11
RATE = 44100
ARRAY_DTYPE = np.int16
sd.default.samplerate = RATE

owner_recordings_dir = Path(__file__).parent.parent / 'owner_recordings'
db_dir = Path(__file__).parent.parent / 'data' / 'dbs'

db_structure = {
    'self': {
        'names': ['id'],
        'types': ['text'],
    },
    'events': {
        'names': ['event_id', 't_start', 't_end', 't_len', 'energy', 'power', 'pressure_mean', 'pressure_sum', 'class', 'audio'],
        'types': ['integer', 'text', 'text', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'text', 'blob'],
    },
    'owner_events': {
        'names': ['t_start', 'response_to', 'name', 'reason', 'sentiment'],
        'types': ['text', 'integer', 'text', 'text', 'text'],
    },
}
