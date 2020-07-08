#! /usr/bin/env python

import maple
import maple.events as events

import time

class RecordOwnerVoice(events.Monitor):
    """Initiate this class to record and store audio clips to yell at your dog"""

    def __init__(self):
        events.Monitor.__init__(self, quiet=True)

        self.menu = {
            'home': {
                'msg': 'Press [r] to record a new sound, [d] to delete all. Press [q] to quit. Response: ',
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

        maple.owner_recordings_dir.mkdir(exist_ok=True)
        self.wav_files = list(maple.owner_recordings_dir.glob('*wav'))
        self.num_recordings = len(self.wav_files)

        print(f"You have {self.num_recordings} recordings.")


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


    def delete_handle(self, response):
        if response == 'y':
            print('Deleted all voice recordings...')
            self.delete_all()
            self.state = 'home'
        elif response == 'n':
            print('Nothing deleted...')
            self.state = 'home'
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def review_handle(self, response):
        if response == 'l':
            print('Played recording...')
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
            self.store(response)
            print('Stored voice input...')
            print(f"You now have {self.num_recordings} recordings.")
            self.state = 'home'


    def store(self, name):
        wav_write(maple.owner_recordings_dir/(name+'.wav'), maple.RATE, self.recording)
        self.num_recordings += 1

