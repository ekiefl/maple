#! /usr/bin/env python

import maple

import time
import gzip
import numpy as np
import datetime

from collections import OrderedDict


def calc_energy(data):
    """Calculate the energy of a discrete time signal"""

    data = data/maple.RATE
    return np.sum(data**2)


def calc_mean_pressure(data):
    """Calculate the mean absolute pressure of a discrete time signal"""

    data = data/maple.RATE
    return np.mean(np.abs(data))


def calc_total_pressure(data):
    """Calculate the sum of absolute pressure of a discrete time signal"""

    data = data/maple.RATE
    return np.sum(np.abs(data))


def convert_array_to_blob(array):
    return gzip.compress(memoryview(array), compresslevel=1)


def convert_blob_to_array(blob, dtype=maple.ARRAY_DTYPE):
    try:
        return np.frombuffer(gzip.decompress(blob), dtype=dtype)
    except:
        return np.frombuffer(blob, dtype=dtype)


class Timer:
    """Manages an ordered dictionary, where each key is a checkpoint name and value is a timestamp"""

    def __init__(self, initial_checkpoint_key=0):
        self.timer_start = self.timestamp()
        self.initial_checkpoint_key = initial_checkpoint_key
        self.last = self.initial_checkpoint_key
        self.checkpoints = OrderedDict([(initial_checkpoint_key, self.timer_start)])
        self.num_checkpoints = 0


    def timestamp(self):
        return datetime.datetime.fromtimestamp(time.time())


    def elapsed_time(self, as_timedelta=False):
        return self.timedelta_to_checkpoint(self.timestamp()).total_seconds()


    def timedelta_to_checkpoint(self, timestamp=None, checkpoint_key=None):
        if not timestamp: timestamp = self.timestamp()
        if not checkpoint_key: checkpoint_key = self.initial_checkpoint_key
        timedelta = timestamp - self.checkpoints[checkpoint_key]
        return timedelta


    def time_between_checkpoints(self, key2, key1, as_timedelta=False):
        """Find time difference between two checkpoints in seconds"""

        time_diff = self.timedelta_to_checkpoint(self.checkpoints[key2], key1)
        return time_diff if as_timedelta else time_diff.total_seconds()


    def make_checkpoint(self, checkpoint_key = None, increment_to = None):
        if not checkpoint_key:
            checkpoint_key = self.num_checkpoints + 1

        if checkpoint_key in self.checkpoints:
            raise ValueError('Timer.make_checkpoint :: %s already exists as a checkpoint key. '
                             'All keys must be unique' % (str(checkpoint_key)))

        checkpoint = self.timestamp()

        self.checkpoints[checkpoint_key] = checkpoint
        self.last = checkpoint_key

        self.num_checkpoints += 1

        return checkpoint
