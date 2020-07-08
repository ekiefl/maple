#! /usr/bin/env python

import maple
import maple.utils as utils

import time
import numpy as np
import pyaudio
import argparse
import datetime
import sounddevice as sd

from collections import OrderedDict


class Detector(object):
    def __init__(self, background_std, background, start_thresh, end_thresh, num_consecutive, seconds, dt, quiet=False):
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

        quiet : bool
            If True, nothing is sent to stdout
        """

        self.quiet = quiet

        self.dt = dt
        self.bg_std = background_std
        self.bg_mean = background
        self.seconds = seconds
        self.num_consecutive = num_consecutive

        # Recast the start and end thresholds in terms of power values
        self.start_thresh = self.bg_mean + start_thresh*self.bg_std
        self.end_thresh = self.bg_mean + end_thresh*self.bg_std

        self.reset()


    def update_event_states(self, power):
        """Update event states based on their current states plus the power of the current frame"""

        if self.in_event:
            if self.in_off_transition:
                if self.off_time > self.seconds:
                    if not self.quiet: print('####### EVENT END #########')
                    self.in_event = False
                    self.in_off_transition = False
                    self.event_finished = True
                elif power > self.start_thresh:
                    self.in_off_transition = False
                else:
                    self.off_time += self.dt
            else:
                if power < self.end_thresh:
                    self.in_off_transition = True
                    self.off_time = 0
                else:
                    pass
        else:
            if self.in_on_transition:
                # Not in event
                if self.on_counter >= self.num_consecutive:
                    if not self.quiet: print('####### EVENT START #########')
                    self.in_event = True
                    self.in_on_transition = False
                elif power > self.start_thresh:
                    self.on_counter += 1
                else:
                    self.in_on_transition = False
                    self.frames = []
            else:
                if power > self.start_thresh:
                    self.in_on_transition = True
                    self.on_counter = 0
                else:
                    # Silence
                    pass


    def print_to_stdout(self):
        """Prints to standard out to create a text-based stream of event detection"""

        if self.quiet:
            return

        if self.in_event:
            if self.in_off_transition:
                msg = '         | '
            else:
                msg = '        |||'
        else:
            if self.in_on_transition:
                msg = '         | '
            else:
                msg = ''

        print(msg)


    def reset(self):
        """Reset event states and storage buffer"""

        self.in_event = False
        self.in_off_transition = False
        self.in_on_transition = False
        self.event_finished = False

        self.frames = []


    def append_to_buffer(self, data):
        if self.in_event or self.in_on_transition:
            self.frames.append(data)


    def process(self, data):
        """Takes in data and updates event transition variables if need be"""

        # Calculate power of frame
        power = utils.calc_power(data)

        self.update_event_states(power)

        # Write to stdout if not self.quiet
        self.print_to_stdout()

        # Append to buffer
        self.append_to_buffer(data)


    def get_event_data(self):
        return np.concatenate(self.frames)


class Monitor(object):
    def __init__(self, args = argparse.Namespace()):
        self.args = args

        self.p = pyaudio.PyAudio()
        self.detector = None

        self.dt = maple.CHUNK/maple.RATE # Time between each sample

        # Calibration parameters
        self.calibration_time = 3 # How many seconds is calibration window
        self.calibration_threshold = 0.50 # Required ratio of std power to mean power
        self.calibration_tries = 1 # Number of running windows tried until threshold is doubled

        # Event detection parameters
        self.event_start_threshold = 3 # standard deviations above background noise to start an event
        self.event_end_threshold = 2 # standard deviations above background noise to end an event
        self.seconds = 0.5
        self.num_consecutive = 5

        self.timer = None
        self.stream = None
        self.background = None
        self.background_std = None

        self.events = {}
        self.num_events = 0


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
            try:
                self.process_data(self.read_chunk())
            except KeyboardInterrupt:
                break

        for key, event in self.events.items():
            print(f"{self.timer.checkpoints[(key, 0)]} -> {self.timer.checkpoints[(key, 1)]}")
            sd.play(event, blocking=True)


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
        prior_state = self.detector.in_event

        self.detector.process(data)

        if self.detector.in_event and not prior_state:
            # Make checkpoint for start of event
            self.timer.make_checkpoint((self.num_events, 0))

        elif self.detector.event_finished:
            # Make checkpoint for end of event
            self.timer.make_checkpoint((self.num_events, 1))

            self.events[self.num_events] = self.detector.get_event_data()

            self.num_events += 1
            self.detector.reset()


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
