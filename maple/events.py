#! /usr/bin/env python

import maple
import maple.utils as utils
import maple.audio as audio

from maple.owner_recordings import OwnerRecordings

import time
import numpy as np
import pandas as pd
import pyaudio
import argparse
import datetime
import sounddevice as sd

from pathlib import Path
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write


class Stream(object):
    def __init__(self):
        self.p = pyaudio.PyAudio()

        self._stream = self.p.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = maple.RATE,
            input = True,
            frames_per_buffer = maple.CHUNK,
            start = False, # To read from stream, self.stream.start_stream must be called
        )


    def __enter__(self):
        if not self._stream.is_active():
            self._stream.start_stream()

        return self


    def __exit__(self, exc_type, exc_val, traceback):
        self._stream.stop_stream()


    def close(self):
        """Close the stream gracefully"""

        if self._stream.is_active():
            self._stream.stop_stream()
        self._stream.close()
        self.p.terminate()


class Detector(object):
    def __init__(self, background_std, background, start_thresh, end_thresh, num_consecutive, seconds, dt, hang_time, wait_timeout, quiet=True):
        """Manages the detection of events

        Parameters
        ==========
        background_std : float
            The standard deviation of the background noise.

        background : float
            The mean of the background noise.

        start_thresh : float
            The number of standard deviations above the background noise that the pressure must exceed
            for a data point to be considered as the start of an event.

        end_thresh : float
            The number of standard deviations above the background noise that the pressure dip below
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

        hang_time : float
            If an event lasts this long (seconds), the flag self.hang is set to True

        wait_timeout : float
            If no event occurs in this amount of time (seconds), self.timeout is set to True

        quiet : bool
            If True, nothing is sent to stdout
        """

        self.quiet = quiet

        self.dt = dt
        self.bg_std = background_std
        self.bg_mean = background
        self.seconds = seconds
        self.num_consecutive = num_consecutive

        # Recast the start and end thresholds in terms of pressure values
        self.start_thresh = self.bg_mean + start_thresh*self.bg_std
        self.end_thresh = self.bg_mean + end_thresh*self.bg_std

        self.hang_time = datetime.timedelta(seconds=hang_time)
        self.wait_timeout = datetime.timedelta(seconds=wait_timeout)

        self.reset()


    def update_event_states(self, pressure):
        """Update event states based on their current states plus the pressure of the current frame"""

        if self.in_event and self.timer.timedelta_to_checkpoint(checkpoint_key='start') > self.hang_time:
            # Event has lasted more than self.hang_time seconds
            self.hang = True

        if not self.in_event and self.timer.timedelta_to_checkpoint(checkpoint_key=0) > self.wait_timeout:
            self.timeout = True

        if self.event_started:
            self.event_started = False

        if self.in_event:
            if self.in_off_transition:
                if self.off_time > self.seconds:
                    self.in_event = False
                    self.in_off_transition = False
                    self.event_finished = True
                elif pressure > self.start_thresh:
                    self.in_off_transition = False
                else:
                    self.off_time += self.dt
            else:
                if pressure < self.end_thresh:
                    self.in_off_transition = True
                    self.off_time = 0
                else:
                    pass
        else:
            if self.in_on_transition:
                # Not in event
                if self.on_counter >= self.num_consecutive:
                    self.in_event = True
                    self.in_on_transition = False
                    self.event_started = True
                elif pressure > self.start_thresh:
                    self.on_counter += 1
                else:
                    self.in_on_transition = False
                    self.frames = []
            else:
                if pressure > self.start_thresh:
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

        if self.event_started:
            msg = '####### EVENT START #########'
        elif self.event_finished:
            msg = '####### EVENT END #########'
        print(msg)


    def reset(self):
        """Reset event states and storage buffer"""

        self.in_event = False
        self.in_off_transition = False
        self.in_on_transition = False
        self.event_finished = False
        self.event_started = False
        self.hang = False
        self.timeout = False

        self.timer = utils.Timer()
        self.frames = []


    def append_to_buffer(self, data):
        if self.in_event or self.in_on_transition:
            self.frames.append(data)


    def process(self, data):
        """Takes in data and updates event transition variables if need be"""

        # Calculate pressure of frame
        pressure = utils.calc_mean_pressure(data)

        self.update_event_states(pressure)

        if self.event_started:
            self.timer.make_checkpoint('start')
        elif self.event_finished:
            self.timer.make_checkpoint('finish')

        # Write to stdout if not self.quiet
        self.print_to_stdout()

        # Append to buffer
        self.append_to_buffer(data)


    def get_event_data(self):
        return np.concatenate(self.frames)


class Monitor(object):
    def __init__(self, args = argparse.Namespace()):

        self.args = args

        A = lambda x: self.args.__dict__.get(x, None)
        self.quiet = A('quiet') or False
        self.calibration_time = A('calibration_time') or 3 # How many seconds is calibration window
        self.calibration_threshold = A('calibration_threshold') or 0.3 # Required ratio of std pressure to mean pressure
        self.calibration_tries = A('calibration_tries') or 4 # Number of running windows tried until threshold is doubled
        self.event_start_threshold = A('event_start_threshold') or 4 # standard deviations above background noise to start an event
        self.event_end_threshold = A('event_end_threshold') or 4 # standard deviations above background noise to end an event
        self.seconds = A('seconds') or 0.25 # see Detector docstring
        self.num_consecutive = A('num_consecutive') or 4 # see Detector docstring
        self.hang_time = A('hang_time') or 20 # see Detector docstring
        self.wait_timeout = A('wait_timeout') or 10 # see Detector docstring

        self.stream = None
        self.background = None
        self.background_std = None
        self.background_audio = None

        self.detector = None
        self.event_recs = {}
        self.num_events = 0

        self.dt = maple.CHUNK/maple.RATE # Time between each sample


    def read_chunk(self):
        """Read a chunk from the stream and cast as a numpy array"""

        return np.fromstring(self.stream._stream.read(maple.CHUNK), dtype=maple.ARRAY_DTYPE)


    def calibrate_background_noise(self):
        """Establish a background noise

        Calculates moving windows of pressure. If the ratio between standard deviation and mean is less
        than a threshold, signifying a constant level of noise, the mean pressure is chosen as the
        background. Otherwise, it is tried again. If it fails too many times, the threshold is
        increased and the process is repeated.
        """
        print('Calibrating.')

        pressure_vals = []
        audio = []

        # Number of chunks in running window based on self.calibration_time
        running_avg_domain = int(self.calibration_time / self.dt)

        calibration_thresh = self.calibration_threshold

        with self.stream:
            tries = 0
            while True:
                for i in range(running_avg_domain):
                    data = self.read_chunk()
                    pressure_vals.append(utils.calc_mean_pressure(data))
                    audio.append(data)

                # Test if threshold met
                pressure_vals = np.array(pressure_vals)
                ratio = np.std(pressure_vals)/np.mean(pressure_vals)
                if ratio < calibration_thresh:
                    self.background_audio = np.concatenate(audio)
                    self.background = np.mean(pressure_vals) # same as utils.calc_mean_pressure(self.background_audio)
                    self.background_std = np.std(pressure_vals)
                    self.background_energy = utils.calc_energy(self.background_audio)
                    self.background_power = self.background_energy/self.calibration_time
                    print('Calibrated.')
                    return

                # Threshold not met, try again
                print(f"Failed: Aimed for below {calibration_thresh}, got {ratio:.2f} ... Trying again.")
                pressure_vals = []
                audio = []
                tries += 1

                if tries == self.calibration_tries:
                    # Max tries met--increase calibration threshold
                    print(f'Calibration threshold not met after {tries} tries. Increasing threshold ({calibration_thresh:.2f} --> {0.1 + calibration_thresh:.2f})')
                    tries = 0
                    calibration_thresh += 0.1


    def setup(self):
        self.stream = Stream()
        self.recalibrate()


    def recalibrate(self):
        self.calibrate_background_noise()

        self.detector = Detector(
            background_std = self.background_std,
            background = self.background,
            start_thresh = self.event_start_threshold,
            end_thresh = self.event_end_threshold,
            seconds = self.seconds,
            num_consecutive = self.num_consecutive,
            hang_time = self.hang_time,
            wait_timeout = self.wait_timeout,
            dt = self.dt,
            quiet = self.quiet,
        )


    def wait_for_event(self, timeout=False):
        """Waits for an event

        Records the event, and returns the event audio as numpy array.

        Parameters
        ==========
        timeout : bool, False
            If True, returns None after self.detector.wait_timeout seconds passes without detecting
            the start of an event.
        """

        self.detector.reset()

        with self.stream:
            while True:
                self.detector.process(self.read_chunk())

                if self.detector.event_finished:
                    break

                if self.detector.hang:
                    print('Event hang... Recalibrating')
                    self.recalibrate()
                    return self.wait_for_event()

                if timeout and self.detector.timeout:
                    return None

        return self.detector.get_event_data()


    def stream_pressure_and_pitch_to_stdout(self, data):
        """Call for every chunk to create a primitive stream plot of pressure and pitch to stdout

        Pitch is indicated with 'o' bars, amplitude is indicated with '-'
        """

        pressure = utils.calc_mean_pressure(data)
        bars = "-"*int(1000*pressure/2**16)

        print("%05d %s" % (pressure, bars))

        w = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        peak = abs(freqs[np.argmax(w)] * maple.RATE)
        bars="o"*int(3000*peak/2**16)

        print("%05d %s" % (peak, bars))


class Responder(object):
    def __init__(self, args = argparse.Namespace(quiet=False)):
        A = lambda x: args.__dict__.get(x, None)

        self.praise = A('praise')
        self.praise_max_events = A('praise_max_events') or 10
        self.praise_max_pressure_sum = A('praise_max_pressure_sum') or 0.01
        self.praise_response_window = A('praise_response_window') or 2
        self.praise_cooldown = A('praise_cooldown') or 2

        self.scold = A('scold')
        self.scold_threshold = A('scold_threshold') or 0.7
        self.scold_trigger = A('scold_trigger') or 0.03
        self.scold_response_window = A('scold_response_window') or 0.5
        self.scold_cooldown = A('scold_cooldown') or 5

        # FIXME not implemented
        self.warn = A('warn')
        self.warn_response_window = A('warn_response_window') or 0.25
        self.warn_cooldown = A('warn_cooldown') or 1

        if self.praise is None:
            self.praise = False
        if self.scold is None:
            self.scold = False
        if self.warn is None:
            self.warn = False

        self.should_respond = A('should_respond')

        # Cast everything as datetime
        self.response_window = datetime.timedelta(minutes=max([
            self.warn_response_window if self.warn else 0,
            self.scold_response_window if self.scold else 0,
            self.praise_response_window if self.praise else 0,
        ]))

        self.warn_response_window = datetime.timedelta(minutes=self.warn_response_window)
        self.scold_response_window = datetime.timedelta(minutes=self.scold_response_window)
        self.praise_response_window = datetime.timedelta(minutes=self.praise_response_window)
        self.warn_cooldown = datetime.timedelta(minutes=self.warn_cooldown)
        self.scold_cooldown = datetime.timedelta(minutes=self.scold_cooldown)
        self.praise_cooldown = datetime.timedelta(minutes=self.praise_cooldown)

        self.owner = OwnerRecordings()
        self.owner.load()

        self.events_in_window = pd.DataFrame({}, columns=maple.db_structure['events']['names'])
        self.timer = utils.Timer()
        self.timer.make_checkpoint('good') # start a cooldown for praise


    def update_window(self, event=None):
        """Add an event to the data window and remove events outside the response window time

        Parameters
        ==========
        event : dict, None
            A dictionary with keys equal to maple.db_structure['events']['names']
        """

        if event is not None:
            self.add_event(event)

        self.events_in_window = self.events_in_window[(self.timer.timestamp() - self.events_in_window['t_end']) < self.response_window]


    def add_event(self, event):
        self.events_in_window = self.events_in_window.append(event, ignore_index=True)


    def respond(self, sentiment, reason):
        """Play owner recording and return an event dictionary"""

        print(f"Playing owner response: {sentiment}, {reason}")
        name = self.owner.play_random(blocking=True, sentiment=sentiment)
        self.timer.make_checkpoint(sentiment, overwrite=True)

        response_to = self.events_in_window['event_id'].iloc[-1] if sentiment != 'good' else -1

        owner_event = {
            't_start': self.timer.checkpoints[sentiment],
            'name': name,
            'response_to': response_to,
            'reason': reason,
            'sentiment': sentiment,
        }


        return owner_event


    def potentially_respond(self, event):
        if not self.should_respond:
            return None

        self.update_window(event)
        timestamp = self.timer.timestamp()

        should_scold, scold_reason = self.should_scold(timestamp)
        should_praise, praise_reason = self.should_praise(timestamp)

        if should_scold:
            owner_event = self.respond(sentiment='bad', reason=scold_reason)
        elif should_praise:
            owner_event = self.respond(sentiment='good', reason=praise_reason)
        else:
            owner_event = None

        return owner_event


    def should_praise(self, timestamp):
        """Return whether dog should be praised, and the reason"""

        if not self.praise:
            return False, None
        elif self.timer.timedelta_to_checkpoint(timestamp, 'good') < self.praise_cooldown:
            # In praise cooldown
            return False, None

        praise_window = self.events_in_window[(timestamp - self.events_in_window['t_end']) < self.praise_response_window]

        if praise_window.empty:
            return True, 'quiet'

        if praise_window.shape[0] <= self.praise_max_events and praise_window['pressure_sum'].max() <= self.praise_max_pressure_sum:
            return True, 'quiet'

        return False, None


    def should_scold(self, timestamp):
        """Return whether dog should be scolded, and the reason"""

        if not self.scold:
            return False, None
        if self.timer.timedelta_to_checkpoint(timestamp, 'bad') < self.scold_cooldown:
            # In scold cooldown
            return False, None

        scold_window = self.events_in_window[(timestamp - self.events_in_window['t_end']) < self.scold_response_window]

        if scold_window.empty:
            # There are no events so nothing to scold
            return False, None

        if scold_window['pressure_sum'].sum() > self.scold_threshold and scold_window.iloc[-1]['pressure_sum'] > self.scold_trigger:
            return True, 'too_loud'

        return False, None


