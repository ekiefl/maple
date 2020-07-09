#! /usr/bin/env python

import maple
import maple.audio as audio
import maple.events as events

from maple.data import DataBase
from maple.owner_recordings import OwnerRecordings

import time
import pandas as pd
import argparse
import sounddevice as sd


class MonitorDog(events.Monitor):
    """Monitor your dog"""

    def __init__(self, args = argparse.Namespace()):
        events.Monitor.__init__(self, args)

        self.db = DataBase(new_database=True)
        self.cols = maple.db_structure['events']['names']
        self.events = pd.DataFrame({}, columns=self.cols)

        self.buffer_size = 0
        self.max_buffer_size = 1


    def store_buffer(self):
        self.db.insert_rows_from_dataframe('events', self.events)
        self.events = pd.DataFrame({}, columns=self.cols)
        self.buffer_size = 0


    def add_event(self, data):
        """Add event to self.events, taking the event audio (numpy array) as input"""

        event = {
            't_start': self.detector.timer.checkpoints['start'],
            't_end': self.detector.timer.checkpoints['finish'],
            't_len': self.detector.timer.time_between_checkpoints('finish', 'start'),
        }

        self.events = self.events.append(event, ignore_index=True)

        self.buffer_size += 1
        if self.buffer_size == self.max_buffer_size:
            self.store_buffer()


    def run(self):
        self.setup()

        while True:
            self.add_event(self.wait_for_event())





class RecordOwnerVoice(events.Monitor):
    """Record and store audio clips to yell at your dog"""

    def __init__(self):
        events.Monitor.__init__(self, quiet=True)

        self.menu = {
            'home': {
                'msg': 'Press [r] to record a new sound, Press [q] to quit. Response: ',
                'function': self.menu_handle,
            },
            'review': {
                'msg': 'Recording finished. [l] to listen, [r] to retry, [k] to keep. Press [q] to quit. Response: ',
                'function': self.review_handle
            },
            'name': {
                'msg': 'Great! type a name for your audio file (just a name, no extension). Response: ',
                'function': self.name_handle
            },
        }

        self.state = 'home'
        self.recording = None

        self.recs = OwnerRecordings()
        print(f"You have {self.recs.num} recordings.")


    def run(self):
        print('Please be quiet. Establishing background noise...')
        time.sleep(1)
        self.setup()

        while True:
            self.menu[self.state]['function'](input(self.menu[self.state]['msg']))
            print()

            if self.state == 'done':
                print('Bye.')
                break


    def menu_handle(self, response):
        if response == 'r':
            print('Listening for voice input...')
            self.recording = self.wait_for_event()
            self.state = 'review'
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def review_handle(self, response):
        if response == 'l':
            print('Played recording...')
            print(self.recording.dtype)
            self.recording = audio.denoise(self.recording, self.background_audio)
            self.recording = audio.bandpass(self.recording, 150, 20000)
            sd.play(self.recording, blocking=True)
        elif response == 'r':
            print('Listening for voice input...')
            self.recording = self.wait_for_event()
        elif response == 'k':
            self.state = 'name'
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def name_handle(self, response):
        if response == '':
            print('Try again.')
        elif ' ' in response:
            print('No spaces are allowed. Try again.')
        elif response == 'q':
            self.state = 'done'
        else:
            self.recs.write(response, self.recording)
            print('Stored voice input...')
            print(f"You now have {self.num_recordings} recordings.")
            self.state = 'home'


