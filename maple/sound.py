#! /usr/bin/env python

import time
import maple
import numpy as np
import pyaudio
import argparse
import datetime

from collections import OrderedDict


class LiveStream(object):
    def __init__(self, args = argparse.Namespace()):
        self.args = args

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.timer = None


    def start(self):
        while True:
            data = np.fromstring(self.stream.read(maple.CHUNK), dtype=np.int16)
            self.process_data(data)


    def __enter__(self):
        self.timer = Timer()
        self.stream = self.p.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = maple.RATE,
            input = True,
            frames_per_buffer = maple.CHUNK,
        )

        return self


    def __exit__(self, exception_type, exception_value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


    def process_data(self, data):
        peak=np.average(np.abs(data))*2
        bars="#"*int(2000*peak/2**16)
        print("%05d %s"%(peak,bars))
        print(self.timer.time_elapsed())

        x = data

        w = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x))

        #max_freq = abs(freqs[np.argmax(w)] * maple.RATE)
        #peak = max_freq
        #bars="o"*int(6000*peak/2**16)
        #print("%05d %s"%(peak,bars))


class Timer:
    """Manages an ordered dictionary, where each key is a checkpoint name and value is a timestamp.

    Examples
    ========

    >>> import time
    >>> t = Timer(); time.sleep(1)
    >>> t.make_checkpoint('checkpoint_name'); time.sleep(1)
    >>> timedelta = t.timedelta_to_checkpoint(timestamp=t.timestamp(), checkpoint_key='checkpoint_name')
    >>> print(t.format_time(timedelta, fmt = '{days} days, {hours} hours, {seconds} seconds', zero_padding=0))
    >>> print(t.time_elapsed())
    0 days, 0 hours, 1 seconds
    00:00:02

    >>> t = Timer(3) # 3 checkpoints expected until completion
    >>> for _ in range(3):
    >>>     time.sleep(1); t.make_checkpoint()
    >>>     print('complete: %s' % t.complete)
    >>>     print(t.eta(fmt='ETA: {seconds} seconds'))
    complete: False
    ETA: 02 seconds
    complete: False
    ETA: 01 seconds
    complete: True
    ETA: 00 seconds
    """
    def __init__(self, required_completion_score=None, initial_checkpoint_key=0, score=0):
        self.timer_start = self.timestamp()
        self.initial_checkpoint_key = initial_checkpoint_key
        self.last_checkpoint_key = self.initial_checkpoint_key
        self.checkpoints = OrderedDict([(initial_checkpoint_key, self.timer_start)])
        self.num_checkpoints = 0

        self.required_completion_score = required_completion_score
        self.score = score
        self.complete = False

        self.last_eta = None
        self.last_eta_timestamp = self.timer_start

        self.scores = {self.initial_checkpoint_key: self.score}


    def timestamp(self):
        return datetime.datetime.fromtimestamp(time.time())


    def timedelta_to_checkpoint(self, timestamp, checkpoint_key=None):
        if not checkpoint_key: checkpoint_key = self.initial_checkpoint_key
        timedelta = timestamp - self.checkpoints[checkpoint_key]
        return timedelta


    def make_checkpoint(self, checkpoint_key = None, increment_to = None):
        if not checkpoint_key:
            checkpoint_key = self.num_checkpoints + 1

        if checkpoint_key in self.checkpoints:
            raise TerminalError('Timer.make_checkpoint :: %s already exists as a checkpoint key. '
                                'All keys must be unique' % (str(checkpoint_key)))

        checkpoint = self.timestamp()

        self.checkpoints[checkpoint_key] = checkpoint
        self.last_checkpoint_key = checkpoint_key

        self.num_checkpoints += 1

        if increment_to:
            self.score = increment_to
        else:
            self.score += 1

        self.scores[checkpoint_key] = self.score

        if self.required_completion_score and self.score >= self.required_completion_score:
            self.complete = True

        return checkpoint


    def gen_dataframe_report(self):
        """Returns a dataframe"""

        d = {'key': [], 'time': [], 'score': []}
        for checkpoint_key, checkpoint in self.checkpoints.items():
            d['key'].append(checkpoint_key)
            d['time'].append(checkpoint)
            d['score'].append(self.scores[checkpoint_key])

        return pd.DataFrame(d)


    def time_elapsed(self, fmt=None):
        return self.format_time(self.timedelta_to_checkpoint(self.timestamp(), checkpoint_key = 0), fmt=fmt)


    def format_time(self, timedelta, fmt = '{hours}:{minutes}:{seconds}', zero_padding = 2):
        """Formats time

        Examples of `fmt`. Suppose the timedelta is seconds = 1, minutes = 1, hours = 1.

            {hours}h {minutes}m {seconds}s  --> 01h 01m 01s
            {seconds} seconds               --> 3661 seconds
            {weeks} weeks {minutes} minutes --> 0 weeks 61 minutes
            {hours}h {seconds}s             --> 1h 61s
        """

        unit_hierarchy = ['seconds', 'minutes', 'hours', 'days', 'weeks']
        unit_denominations = {'weeks': 7, 'days': 24, 'hours': 60, 'minutes': 60, 'seconds': 1}

        if not fmt:
            # use the highest two non-zero units, e.g. if it is 7200s, use {hours}h{minutes}m
            seconds = int(timedelta.total_seconds())
            if seconds < 60:
                fmt = '{seconds}s'
            else:
                m = 1
                for i, unit in enumerate(unit_hierarchy):
                    if not seconds // (m * unit_denominations[unit]) >= 1:
                        fmt = '{%s}%s{%s}%s' % (unit_hierarchy[i-1],
                                                unit_hierarchy[i-1][0],
                                                unit_hierarchy[i-2],
                                                unit_hierarchy[i-2][0])
                        break
                    elif unit == unit_hierarchy[-1]:
                        fmt = '{%s}%s{%s}%s' % (unit_hierarchy[i],
                                                unit_hierarchy[i][0],
                                                unit_hierarchy[i-1],
                                                unit_hierarchy[i-1][0])
                        break
                    else:
                        m *= unit_denominations[unit]

        # parse units present in fmt
        format_order = []
        for i, x in enumerate(fmt):
            if x == '{':
                for j, k in enumerate(fmt[i:]):
                    if k == '}':
                        unit = fmt[i+1:i+j]
                        format_order.append(unit)
                        break

        if not format_order:
            raise TerminalError('Timer.format_time :: fmt = \'%s\' contains no time units.' % (fmt))

        for unit in format_order:
            if unit not in unit_hierarchy:
                raise TerminalError('Timer.format_time :: \'%s\' is not a valid unit. Use any of %s.'\
                                     % (unit, ', '.join(unit_hierarchy)))

        # calculate the value for each unit (e.g. 'seconds', 'days', etc) found in fmt
        format_values_dict = {}
        smallest_unit = unit_hierarchy[[unit in format_order for unit in unit_hierarchy].index(True)]
        units_less_than_or_equal_to_smallest_unit = unit_hierarchy[::-1][unit_hierarchy[::-1].index(smallest_unit):]
        seconds_in_base_unit = 1
        for a in [v for k, v in unit_denominations.items() if k in units_less_than_or_equal_to_smallest_unit]:
            seconds_in_base_unit *= a
        r = int(timedelta.total_seconds()) // seconds_in_base_unit

        for i, lower_unit in enumerate(unit_hierarchy):
            if lower_unit in format_order:
                m = 1
                for upper_unit in unit_hierarchy[i+1:]:
                    m *= unit_denominations[upper_unit]
                    if upper_unit in format_order:
                        format_values_dict[upper_unit], format_values_dict[lower_unit] = divmod(r, m)
                        break
                else:
                    format_values_dict[lower_unit] = r
                    break
                r = format_values_dict[upper_unit]

        format_values = [format_values_dict[unit] for unit in format_order]

        style_str = '0' + str(zero_padding) if zero_padding else ''
        for unit in format_order:
            fmt = fmt.replace('{%s}' % unit, '%' + '%s' % (style_str) + 'd')
        formatted_time = fmt % (*[format_value for format_value in format_values],)

        return formatted_time
