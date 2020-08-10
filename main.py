#! /usr/bin/env python

import numpy as np
import pyaudio
import argparse

CHUNK = 2**11
RATE = 44100


def calc_power(data):
    """Calculate the power of a discrete time signal"""

    return np.mean(np.abs(data))*2


class Stream(object):
    def __init__(self):
        self.p = pyaudio.PyAudio()

        self._stream = self.p.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = RATE,
            input = True,
            frames_per_buffer = CHUNK,
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


class Monitor(object):
    def __init__(self, args = argparse.Namespace()):
        self.args = args

        self.dt = CHUNK/RATE # Time between each sample

        # Calibration parameters
        self.calibration_time = 3 # How many seconds is calibration window
        self.calibration_threshold = 0.50 # Required ratio of std power to mean power
        self.calibration_tries = 1 # Number of running windows tried until threshold is doubled

        # Event detection parameters
        self.event_start_threshold = 3 # standard deviations above background noise to start an event
        self.event_end_threshold = 2 # standard deviations above background noise to end an event
        self.seconds = 0.5
        self.num_consecutive = 4

        self.stream = None
        self.background = None
        self.background_std = None

        self.detector = None
        self.event_recs = {}
        self.num_events = 0


    def read_chunk(self):
        """Read a chunk from the stream and cast as a numpy array"""

        return np.fromstring(self.stream._stream.read(CHUNK), dtype=np.int16)


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

        with self.stream:
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


    def setup(self):
        self.stream = Stream()

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

        self.wait_for_event()


    def wait_for_event(self, log=True):
        """Waits for an event, records the event, and returns the event audio as numpy array"""

        self.detector.reset()

        with self.stream:
            while True:
                self.detector.process(self.read_chunk())

                if self.detector.event_finished:
                    break

        return self.detector.get_event_data()


    def stream_power_and_pitch_to_stdout(self, data):
        """Call for every chunk to create a primitive stream plot of power and pitch to stdout

        Pitch is indicated with 'o' bars, amplitude is indicated with '-'
        """

        power = utils.calc_power(data)
        bars = "-"*int(1000*power/2**16)

        print("%05d %s" % (power, bars))

        w = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        peak = abs(freqs[np.argmax(w)] * RATE)
        bars="o"*int(3000*peak/2**16)

        print("%05d %s" % (peak, bars))


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

        if self.event_started:
            self.event_started = False

        if self.in_event:
            if self.in_off_transition:
                if self.off_time > self.seconds:
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
                    self.in_event = True
                    self.in_on_transition = False
                    self.event_started = True
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
                msg = '         o '
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


if __name__ == '__main__':
    s = Monitor()
    s.setup()
