#! /usr/bin/env python

import maple
import maple.utils as utils
import maple.audio as audio

import numpy as np
import pandas as pd
import shutil
import sqlite3
import datetime
import sounddevice as sd

from pathlib import Path
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write


class OwnerRecordings(object):
    def __init__(self):
        self.dir = maple.owner_recordings_dir
        self.dir.mkdir(exist_ok=True)

        self.load()


    def write(self, name, data, fs, sentiment=None):
        """Write a numpy array to a .wav file

        Parameters
        ==========
        name : str
            Just the basename of the file, with no extensions

        data : numpy array
            A numpy array

        fs : int
            sampling freq

        sentiment : str, None
            label for the owner recording
        """

        output = self.dir/(name+'.wav')
        wav_write(output, fs, data)

        if self.summary is not None:
            new_summary = self.summary.append({'name': name, 'sentiment': sentiment}, ignore_index=True)
            new_summary.to_csv(self.dir/'summary.txt', sep='\t', index=False)
        else:
            pd.DataFrame({'name': [name], 'sentiment': [sentiment]}).to_csv(self.dir/'summary.txt', sep='\t', index=False)

        self.load()


class DataBase(object):
    def __init__(self, db_path=None, new_database=False, temp=False):
        self.new_database = new_database

        if db_path is None:
            if temp:
                self.db_path = maple.db_dir_temp / self.get_default_db_id() / 'events.db'
            else:
                self.db_path = maple.db_dir / self.get_default_db_id() / 'events.db'
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_id = self.db_path.parent.stem

        if self.new_database and self.db_path.exists():
            raise ValueError(f"db {self.db_path} already exists.")
        elif not self.new_database and not self.db_path.exists:
            raise ValueError(f"db {self.db_path} doesn't exists.")

        self.conn = sqlite3.connect(self.db_path)
        self.conn.text_factory = str
        self.cursor = self.conn.cursor()

        if self.new_database:
            self.create_self()
            self.create_table('events', maple.db_structure['events']['names'], maple.db_structure['events']['types'])
            self.create_table('owner_events', maple.db_structure['owner_events']['names'], maple.db_structure['owner_events']['types'])


    def get_default_db_id(self):
        dt = datetime.datetime.now()
        return '_'.join([f"{x:02d}" for x in [
            dt.date().year,
            dt.date().month,
            dt.date().day,
            dt.time().hour,
            dt.time().minute,
            dt.time().second,
        ]])


    def set_meta_value(self, key, value):
        self.remove_meta_key_value_pair(key)
        self._exec('''INSERT INTO self VALUES(?,?)''', (key, value,))
        self.commit()


    def remove_meta_key_value_pair(self, key):
        self._exec('''DELETE FROM self WHERE key="%s"''' % key)
        self.commit()


    def set_meta_values(self, d):
        for k, v in d.items():
            self.set_meta_value(k, v)


    def get_table_as_dataframe(self, table_name, columns_of_interest=None):
        table_structure = self.get_table_structure(table_name)

        if not columns_of_interest:
            columns_of_interest = table_structure

        results_df = pd.read_sql('''SELECT * FROM "%s"''' % table_name, self.conn, columns=table_structure)

        return results_df[columns_of_interest]


    def insert_rows_from_dataframe(self, table_name, df, replace=False):
        df = df[self.get_table_structure(table_name)]

        df.to_sql(
            table_name,
            self.conn,
            if_exists='replace' if replace else 'append',
            dtype=self.get_table_columns_and_types(table_name),
            index=False
        )


    def create_self(self):
        self._exec('''CREATE TABLE self (key text, value text)''')

        self.set_meta_value('id', self.db_id)
        self.set_meta_value('start', str(datetime.datetime.now()))
        self.commit()


    def get_table_structure(self, table_name):
        response = self._exec('''SELECT * FROM %s''' % table_name)
        return [t[0] for t in response.description]


    def get_table_columns_and_types(self, table_name):
        response = self._exec('PRAGMA TABLE_INFO(%s)' % table_name)
        return dict([(t[1], t[2]) for t in response.fetchall()])


    def create_table(self, table_name, fields, types):
        db_fields = ', '.join(['%s %s' % (t[0], t[1]) for t in zip(fields, types)])
        self._exec('''CREATE TABLE %s (%s)''' % (table_name, db_fields))
        self.commit()


    def _exec(self, sql_query, value=None):
        if value:
            ret_val = self.cursor.execute(sql_query, value)
        else:
            ret_val = self.cursor.execute(sql_query)

        self.commit()
        return ret_val


    def commit(self):
        self.conn.commit()


    def disconnect(self):
        """Disconnect and potentially update self table values"""

        if self.new_database:
            self.set_meta_value('end', datetime.datetime.now())

        self.conn.commit()
        self.conn.close()


    def self_destruct(self, rm_dir=False, are_you_sure=False):
        self.disconnect()
        self.db_path.unlink()

        if rm_dir and are_you_sure:
            shutil.rmtree(self.db_path.parent)


class SessionAnalysis(DataBase):
    def __init__(self, name=None, path=None, temp=False):
        if (path and name) or not (path or name):
            raise ValueError("choose one of --path or --name")

        if path:
            self.db_path = path
        else:
            if temp:
                self.db_path = maple.db_dir_temp / name / 'events.db'
            else:
                self.db_path = maple.db_dir / name / 'events.db'

        DataBase.__init__(self, db_path=self.db_path)

        self.get_meta()
        self.get_dog_events()
        self.get_owner_events()

        self.start = datetime.datetime.strptime(self.meta.loc[self.meta['key'] == 'start', 'value'].iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        self.end = datetime.datetime.strptime(self.meta.loc[self.meta['key'] == 'end', 'value'].iloc[0], '%Y-%m-%d %H:%M:%S.%f')
        self.duration = (self.end - self.start).total_seconds()

        self.trim_ends(minutes=1)

        self.num_events = self.dog.shape[0]
        self.psds = None


    def trim_ends(self, minutes=1):
        self.dog = self.dog[
            (self.dog.t_end <= self.end - datetime.timedelta(minutes=minutes)) & \
            (self.dog.t_start >= self.start + datetime.timedelta(minutes=minutes))
        ]


    def calc_PSDs(self):
        self.psds = {
            'event_id': np.array([], dtype=int),
            'freq': np.array([]),
            'psd': np.array([]),
        }

        for event_id, data in self.dog['audio'].items():
            f, psd = audio.PSD(data)
            self.psds['freq'] = np.concatenate((self.psds['freq'], f))
            self.psds['psd'] = np.concatenate((self.psds['psd'], psd))
            self.psds['event_id'] = np.concatenate((self.psds['event_id'], event_id * np.ones(len(psd), dtype=int)))

        self.psds = pd.DataFrame(self.psds)


    def get_meta(self):
        self.meta = self.get_table_as_dataframe('self')


    def get_dog_events(self):
        self.dog = self.get_table_as_dataframe('events')

        self.dog['t_start'] = pd.to_datetime(self.dog['t_start'])
        self.dog['t_end'] = pd.to_datetime(self.dog['t_end'])
        self.dog['audio'] = self.dog['audio'].apply(utils.convert_blob_to_array)
        self.dog.set_index('event_id', drop=False, inplace=True)


    def get_owner_events(self):
        self.owner = self.get_table_as_dataframe('owner_events')


    def get_event_audio(self, event_id):
        return self.dog.loc[event_id, 'audio']


    def get_owner_event(self, event_id):
        raise NotImplementedError("FIXME Not implemented yet")


    def play(self, event_id):
        sd.play(self.get_event_audio(event_id), blocking=True)


    def play_many(self, start=None, stop=None):
        """Play in order of df"""

        if start is None:
            start = 0
        if stop is None:
            stop = self.dog.index[-1]

        for event_id in range(start, stop+1):
            print(f"Playing:\n{self.dog.loc[event_id, [x for x in self.dog.columns if x != 'audio']]}\n")
            self.play(event_id)


    def save_event_as_wav(self, event_id, path):
        wav_write(path, maple.RATE, self.dog.loc[event_id, 'audio'])


