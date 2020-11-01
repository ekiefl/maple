#! /usr/bin/env python

import maple
import maple.data as data
import maple.audio as audio
import maple.utils as utils
import maple.events as events

from maple.data import DataBase, SessionAnalysis
from maple.owner_recordings import OwnerRecordings

import time
import datetime
import pandas as pd
import argparse
import sounddevice as sd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

from pathlib import Path


class MonitorDog(events.Monitor):
    """Monitor your dog or boyfriend who won't stop singing"""

    def __init__(self, args = argparse.Namespace(quiet=True)):

        self.args = args

        A = lambda x: self.args.__dict__.get(x, None)
        self.recalibration_rate = A('recalibration_rate') or 10000 # never recalibrate by default
        self.max_buffer_size = A('max_buffer_size') or 100
        self.temp = A('temp')
        self.praise = A('praise')
        self.scold = A('scold')
        self.sound_check = A('sound_check')

        self.recalibration_rate = datetime.timedelta(minutes=self.recalibration_rate)

        events.Monitor.__init__(self, self.args)

        self.db = DataBase(new_database=True, temp=self.temp)
        self.db.set_meta_values({k: v for k, v in vars(args).items() if k in maple.config_parameters})

        self.event_cols = maple.db_structure['events']['names']
        self.events = pd.DataFrame({}, columns=self.event_cols)

        self.buffer_size = 0
        self.event_id = 0

        self.responder = events.Responder(self.args)
        self.owner_event_cols = maple.db_structure['owner_events']['names']
        self.owner_events = pd.DataFrame({}, columns=self.owner_event_cols)

        self.timer = None


    def store_buffer(self):
        self.db.insert_rows_from_dataframe('events', self.events)
        self.events = pd.DataFrame({}, columns=self.event_cols)

        # Also store any owner responses
        self.db.insert_rows_from_dataframe('owner_events', self.owner_events)
        self.owner_events = pd.DataFrame({}, columns=self.owner_event_cols)

        self.buffer_size = 0


    def add_event(self, data):
        """Add event to self.events, taking the event audio (numpy array) as input"""
        if data is None:
            return None

        t_in_sec = self.detector.timer.time_between_checkpoints('finish', 'start')

        energy = utils.calc_energy(data)
        pressure = utils.calc_mean_pressure(data)

        event = {
            'event_id': self.event_id,
            't_start': self.detector.timer.checkpoints['start'],
            't_end': self.detector.timer.checkpoints['finish'],
            't_len': t_in_sec,
            'energy': energy,
            'power': energy/t_in_sec,
            'pressure_mean': pressure,
            'pressure_sum': pressure*t_in_sec,
            'class': None, # TODO
            'audio': utils.convert_array_to_blob(data),
        }

        self.events = self.events.append(event, ignore_index=True)
        self.buffer_size += 1
        self.event_id += 1

        print(f"Event: ID={event['event_id']}; time={event['t_start']}; length={event['t_len']:.1f}s; pressure_sum={event['pressure_sum']:.5f}")

        return event


    def add_owner_event(self, owner_event):
        if owner_event:
            self.owner_events = self.owner_events.append(owner_event, ignore_index=True)


    def play_sample_until_user_happy(self):
        recs = OwnerRecordings()
        while True:
            if self.praise:
                recs.play_random(blocking=True, sentiment='good')
            if self.scold:
                recs.play_random(blocking=True, sentiment='bad')

            response = input('(r) to replay, (s) to start: ')
            if response == 'r':
                continue
            elif response == 's':
                break


    def run(self):
        if (self.praise or self.scold) and self.sound_check:
            self.play_sample_until_user_happy()

        self.setup()
        self.timer = utils.Timer()
        self.timer.make_checkpoint('calibration') # just calibrated

        try:
            while True:
                event = self.add_event(self.wait_for_event(timeout=True))

                self.add_owner_event(self.responder.potentially_respond(event))

                if self.timer.timedelta_to_checkpoint(checkpoint_key='calibration') > self.recalibration_rate:
                    print('Overdue for calibration. Calibrating...')
                    self.recalibrate()
                    self.timer.make_checkpoint('calibration', overwrite=True)

                if self.buffer_size == self.max_buffer_size:
                    self.store_buffer()

        except KeyboardInterrupt:
            while True:
                keep = input("Do you want to keep this? (y/n)\n")

                if keep == 'y':
                    self.db.set_meta_value('background_mean', self.background)
                    self.db.set_meta_value('background_std', self.background_std)
                    self.store_buffer()
                    self.db.disconnect()

                    break
                elif keep == 'n':
                    self.db.self_destruct()
                    break
                else:
                    print("Invalid option.")


class Analysis(object):
    def __init__(self, args=argparse.ArgumentParser()):
        """FIXME This will analyze across sessions in the future"""

        A = lambda x: args.__dict__.get(x, None)
        self.name = A('session')
        self.path = A('path')
        self.temp = A('temp')
        self.list = A('list')

        self.bin_size = A('bin_size')

        if self.temp:
            self.db_dir = maple.db_dir_temp
        else:
            self.db_dir = maple.db_dir

        self.names = sorted([x.stem for x in self.db_dir.glob('*') if not x.stem.startswith('.')])
        if self.list:
            for name in self.names:
                print(name)
            import sys
            sys.exit()

        if not self.path and not self.name:
            self.name = self.names[-1]

        self.session = data.SessionAnalysis(self.name, self.path, temp=self.temp)
        self.session.calc_PSDs()


    def run(self):
        self.histogram(self.session)

        # Enter interactive session
        import ipdb; ipdb.set_trace()


    def psd(self, session, ids=None):
        if ids is not None:
            df = session.psds[session.psds['event_id'].isin(ids)]
        else:
            df = session.psds

        fig = px.line(df, x='freq', y='psd', color='event_id')
        fig.update_layout(yaxis_type='log')
        fig.show()


    def histogram(self, session):
        dog_hover_cols = ['t_start', 'event_id', 't_len', 'energy', 'power', 'pressure_mean', 'pressure_sum']
        owner_hover_cols = ['t_start', 'response_to', 'name', 'reason', 'sentiment']

        color='#ADB9CB'
        color2='#ED028C' # scold
        color3='#01C702' # praise

        get_dog_event_string = lambda row: tabulate(pd.DataFrame(row[row.index.isin(dog_hover_cols)]), tablefmt='fancy_grid').replace('\n', '<br>')
        get_owner_event_string = lambda row: tabulate(pd.DataFrame(row[row.index.isin(owner_hover_cols)]), tablefmt='fancy_grid').replace('\n', '<br>')

        hover_data_dog = session.dog.apply(get_dog_event_string, axis=1).tolist()
        hover_data_owner = session.owner.apply(get_owner_event_string, axis=1).tolist()

        histogram = go.Histogram(
            x=session.dog["t_start"],
            y=session.dog["pressure_sum"],
            histfunc="sum",
            xbins=dict(
                start=session.dog["t_start"].iloc[0],
                end=session.dog["t_start"].iloc[-1],
                size=self.bin_size*10**3
            ),
            name='Sound pressure',
            marker={'color': color},
            showlegend = False,
        )

        p = session.dog['pressure_sum']
        p = 50 * (p - p.min()) / (p.max() - p.min()) + 2.5

        n_dog_events = len(session.dog['t_start'])
        n_owner_events = len(session.owner['t_start'])
        n_owner_praise = session.owner[session.owner['sentiment'] == 'good'].shape[0]
        n_owner_scold = session.owner[session.owner['sentiment'] == 'bad'].shape[0]

        data = session.dog["t_start"].append(session.owner.loc[session.owner['sentiment']=='bad', 't_start'])
        data = data.append(session.owner.loc[session.owner['sentiment']=='good', 't_start'])

        dog_rug = dict(
            mode = "markers",
            name = "Events",
            type = "scatter",
            x = data,
            y = ['Event'] * n_dog_events + ['Event'] * n_owner_events,
            marker = dict(
                color = [color] * n_dog_events + [color2] * n_owner_scold + [color3] * n_owner_praise,
                symbol = "line-ns-open",
                size = p.tolist() + [20] * n_owner_events,
                line = dict(
                    width = [1] * n_dog_events + [2] * n_owner_events,
                )
            ),
            showlegend = False,
            text = hover_data_dog + hover_data_owner,
            hoverinfo = 'text',
            hoverlabel = dict(
                font = dict(
                    family='courier new',
                    size=8,
                ),
                bgcolor='white',
            ),
        )

        fig = go.Figure()
        fig = make_subplots(rows=2, cols=1, row_width=[0.2, 0.8], shared_xaxes=True)
        fig.add_trace(dog_rug, 2, 1)
        fig.add_trace(histogram, 1, 1)
        fig.update_layout(
            template='none',
            title="Barking distribution",
            title_font_family="rockwell",
            font_family="rockwell",
            yaxis_title="Sound pressure [AU]",
        )
        fig.update_xaxes(matches='x')
        fig.show()

        self.save_fig(fig, session.db_path.parent / 'histogram.html')


    def save_fig(self, fig, path):
        with open(path, 'w') as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))


class RecordOwnerVoice(events.Monitor):
    """Record and store audio clips to yell at your dog"""

    def __init__(self, args=argparse.Namespace(quiet=True)):
        events.Monitor.__init__(self, args)

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
            'sentiment': {
                'msg': 'Final question. Choose the sentiment: [g] for good, [b] for bad, [w] for warn. Press [q] to quit. Response: ',
                'function': self.sentiment_handle
            },
        }

        self.state = 'home'
        self.recording = None

        self.recs = OwnerRecordings()
        print(f"You have {self.recs.num} recordings.")


    def run(self):
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
            self.name = response
            self.state = 'sentiment'


    def sentiment_handle(self, response):
        if response == 'w':
            sentiment = 'warn'
        elif response == 'g':
            sentiment = 'good'
        elif response == 'b':
            sentiment = 'bad'
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')
            return

        self.recs.write(self.name, self.recording, maple.RATE, sentiment)
        print('Stored voice input...')
        print(f"You now have {self.recs.num} recordings.")
        self.state = 'home'




