#! /usr/bin/env python

import ast
import numpy as np
import sounddevice as sd
import configparser

from pathlib import Path

CHUNK = 2**11
RATE = 44100
ARRAY_DTYPE = np.int16
sd.default.samplerate = RATE

owner_recordings_dir = Path(__file__).parent.parent / 'data' / 'owner_recordings'
db_dir = Path(__file__).parent.parent / 'data' / 'sessions'
db_dir_temp = Path(__file__).parent.parent / 'data' / 'temp'

# Load up the configuration file, store as nested dictionary `config`
config_path = Path(__file__).parent.parent / 'config'
config_obj = configparser.ConfigParser()
config_obj.read(config_path)
config = {}
config_parameters = []
for section in config_obj.sections():
    config[section] = {}
    for k, v in config_obj[section].items():
        config_parameters.append(k)
        try:
            config[section][k] = ast.literal_eval(v)
        except:
            config[section][k] = v

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

