#! /usr/bin/env python

import maple
import maple.utils as utils

import time
import numpy as np
import pyaudio
import argparse
import datetime

from collections import OrderedDict


class Detector(object):
    def __init__(self, background_std, background, start_thresh, end_thresh, num_consecutive, seconds, dt):
        """Manages the detection of events

        Parameters
        ==========
        background_std : float
            The standard deviation of the background noise.

        background : float
            The mean of the background noise.

        start_thresh : float
            The number of standard deviations above the background noise that the power must exceed
            for a data point to be considered as the start of an event.

        end_thresh : float
            The number of standard deviations above the background noise that the power dip below
            for a data point to be considered as the end of an event.

        num_consecutive : int
            The number of frames needed that must consecutively be above the threshold to be
            considered the start of an event.

        seconds : float
            The number of seconds that must pass after the `end_thresh` condition is met in order
            for the event to end. If, during this time, the `start_thresh` condition is met, the
            ending of the event will be cancelled.

        dt : float
            The inverse sampling frequency, i.e the time captured by each frame.
        """

        self.dt = dt
        self.bg_std = background_std
        self.bg_mean = background
        self.seconds = seconds
        self.num_consecutive = num_consecutive

        # Recast the start and end thresholds in terms of power values
        self.start_thresh = self.bg_mean + start_thresh*self.bg_std
        self.end_thresh = self.bg_mean + end_thresh*self.bg_std

        self.in_event = False

        self.frames = []


    def process(self, data):
        """Takes in data and updates event transition variables if need be"""

        power = utils.calc_power(data)

        if self.in_event:
            if self.in_off_transition:
                if self.off_time > self.seconds:
                    print('####### EVENT END #########')
                    self.in_event = False
                    self.in_off_transition = False
                elif power > self.start_thresh:
                    self.in_off_transition = False
                else:
                    self.off_time += self.dt
            else:
                if power < self.end_thresh:
                    self.in_off_transition = True


            if power < self.end_thresh and not self.in_off_transition:
                self.in_off_transition = True
                self.off_time = 0

            elif self.off_time > self.seconds:

            elif self.in_off_transition:
                self.off_time += self.dt

            else:
                print('          ||')
        else:
            if power > self.start_thresh:
                print('####### EVENT START #########')
                self.in_event = True
            else:
                print()


class Monitor(object):
    def __init__(self, args = argparse.Namespace()):
        self.args = args

        self.p = pyaudio.PyAudio()
        self.detector = None

        self.dt = maple.CHUNK/maple.RATE # Time between each sample

        # Calibration parameters
        self.calibration_time = 3 # How many seconds is calibration window
        self.calibration_threshold = 0.25 # Required ratio of std power to mean power
        self.calibration_tries = 3 # Number of running windows tried until threshold is doubled

        # Event detection parameters
        self.event_start_threshold = 3 # standard deviations above background noise to start an event
        self.event_end_threshold = 2 # standard deviations above background noise to end an event
        self.seconds = 2
        self.num_consecutive = 3

        self.timer = None
        self.stream = None
        self.background = None
        self.background_std = None


    def read_chunk(self):
        """Read a chunk from the stream and cast as a numpy array"""

        return np.fromstring(self.stream.read(maple.CHUNK), dtype=np.int16)


    def calibrate_background_noise(self):
        """Establish a background noise

        Calculates moving windows of power. If the ratio between standard deviation and mean is less
        than a threshold, signifying a constant level of noise, the mean power is chosen as the
        background. Otherwise, it is tried again. If it fails too many times, the threshold is
        increased and the process is repeated.
        """

        stable = False
        power_vals = []

        # Number of chunks in running window based on self.calibration time
        running_avg_domain = int(self.calibration_time / self.dt)

        tries = 0
        while True:
            for i in range(running_avg_domain):
                power_vals.append(utils.calc_power(self.read_chunk()))

            # Test if threshold met
            power_vals = np.array(power_vals)
            if np.std(power_vals)/np.mean(power_vals) < self.calibration_threshold:
                self.background = np.mean(power_vals)
                self.background_std = np.std(power_vals)
                return

            # Threshold not met, try again
            power_vals = []
            tries += 1

            if tries == self.calibration_tries:
                # Max tries met--doubling calibration threshold
                print(f'Calibration threshold not met after {tries} tries. Increasing threshold ({self.calibration_threshold:.2f} --> {1.5*self.calibration_threshold:.2f})')
                tries = 0
                self.calibration_threshold *= 1.5


    def start(self):
        """Start monitoring audio"""

        self.timer = Timer()
        self.stream = self.p.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = maple.RATE,
            input = True,
            frames_per_buffer = maple.CHUNK,
        )

        self.calibrate_background_noise()

        self.detector = Detector(
            background_std = self.background_std,
            background = self.background,
            start_thresh = self.event_start_threshold,
            end_thresh = self.event_end_threshold,
            seconds = self.seconds,
            num_consecutive = self.num_consecutive,
            dt = self.dt,
        )

        while True:
            #self.stream_power_and_pitch_to_stdout(self.read_chunk())
            self.process_data(self.read_chunk())


    def close(self):
        """Close the stream gracefully"""

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


    def stream_power_and_pitch_to_stdout(self, data):
        """Call for every chunk to create a primitive stream plot of power and pitch to stdout

        Pitch is indicated with 'o' bars, amplitude is indicated with '-'
        """

        power = utils.calc_power(data)
        bars = "-"*int(1000*power/2**16)

        print("%05d %s" % (power, bars))

        w = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        peak = abs(freqs[np.argmax(w)] * maple.RATE)
        bars="o"*int(3000*peak/2**16)

        print("%05d %s" % (peak, bars))


    def process_data(self, data):
        self.detector.process(data)



class Timer:
    """Manages an ordered dictionary, where each key is a checkpoint name and value is a timestamp"""

    def __init__(self, initial_checkpoint_key=0):
        self.timer_start = self.timestamp()
        self.initial_checkpoint_key = initial_checkpoint_key
        self.last_checkpoint_key = self.initial_checkpoint_key
        self.checkpoints = OrderedDict([(initial_checkpoint_key, self.timer_start)])
        self.num_checkpoints = 0


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
            raise ValueError('Timer.make_checkpoint :: %s already exists as a checkpoint key. '
                             'All keys must be unique' % (str(checkpoint_key)))

        checkpoint = self.timestamp()

        self.checkpoints[checkpoint_key] = checkpoint
        self.last_checkpoint_key = checkpoint_key

        self.num_checkpoints += 1

        return checkpoint


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
