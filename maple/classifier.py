#! /usr/bin/env python

import maple
import maple.data as data
import maple.audio as audio

import numpy as np
import joblib
import pandas as pd
import argparse
import datetime
import sounddevice as sd

from scipy import signal
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

labels = {
    0: 'none',
    1: 'whine',
    2: 'howl',
    3: 'bark',
    4: 'play',
    5: 'scratch_cage',
    6: 'scratch_door',
}


class LabelAudio(object):
    """Label audio from Audio"""

    def __init__(self, args=argparse.Namespace()):
        self.menu = {
            'home': {
                'msg': 'Press [s] to start, Press [q] to quit. Response: ',
                'function': self.menu_handle,
            },
            'preview': {
                'msg': 'Use clip? [y]es, [n]o, [r]epeat, [q]uit. Response: ',
                'function': self.preview_handle,
            },
            'label': {
                'msg': '[0] none, [1] whine, [2] howl, [3] bark, [4] play, [5] cage, [6] door, [r]epeat, [R]epeat full, [s] to skip, [q]uit. Reponse: '.\
                    format(', '.join(['[' + str(key) + '] ' + val for key, val in labels.items()])),
                'function': self.label_handle,
            },
        }

        if args.label_data is None:
            raise Exception("You must supply --label-data")

        self.state = 'home'
        self.subevent_time = 0.25 # in seconds

        self.cols = [
            'session_id',
            'event_id',
            'subevent_id',
            't_start',
            't_end',
            'label',
            'date_labeled',
        ]

        self.label_data_path = Path(args.label_data)
        if self.label_data_path.exists():
            self.data = pd.read_csv(self.label_data_path, sep='\t')
        else:
            self.data = pd.DataFrame({}, columns=self.cols)

        print(f'You have {self.data.shape[0]} labelled data')

        if args.session_paths is None:
            session_paths = sorted(maple.db_dir.glob('*/events.db'))
        else:
            session_paths = sorted([Path(x.strip()) for x in open(args.session_paths).readlines()])

        self.sessions = {}
        for session_path in session_paths:
            print(f'Loading session path {session_path}')
            session_db = data.SessionAnalysis(path=session_path)
            session_db.trim_ends(minutes=1)

            if session_db.dog.empty:
                continue

            session_id = session_db.meta[session_db.meta['key'] == 'id'].iloc[0]['value']
            session_db.dog['session_id'] = session_id
            self.sessions[session_id] = session_db.dog
            session_db.disconnect()

        self.filter()


    def run(self):
        while True:
            self.menu[self.state]['function'](input(self.menu[self.state]['msg']))
            print()

            if self.state == 'done':
                self.save_data()
                print('Any new data has been saved. Bye.')
                break


    def menu_handle(self, response):
        if response == 's':
            self.event = self.sample_event()
            self.play_event(self.event)
            self.state = 'preview'
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def preview_handle(self, response):
        if response == 'y':
            self.set_subevents(self.event)
            self.play_subevent(self.curr_subevent)
            self.state = 'label'
        elif response == 'n':
            self.event = self.sample_event()
            self.play_event(self.event)
        elif response == 'r':
            self.play_event(self.event)
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def label_handle(self, response):
        if response == 'r':
            self.play_subevent(self.curr_subevent)
        elif response == 'R':
            self.play_event(self.event)
        elif response == 's':
            if not self.increment_subevent():
                print('Finished event')
                self.cache_event_labels()
                self.event = self.sample_event()
                self.play_event(self.event)
                self.state = 'preview'
                return
            self.play_subevent(self.curr_subevent)
        elif response in [str(x) for x in labels.keys()]:
            self.append(response)
            if not self.increment_subevent():
                print('Finished event')
                self.cache_event_labels()
                self.event = self.sample_event()
                self.play_event(self.event)
                self.state = 'preview'
                return
            self.play_subevent(self.curr_subevent)
        elif response == 'q':
            self.state = 'done'
        else:
            print('invalid input')


    def sample_event(self):
        self.event_data = {x: [] for x in self.cols}
        while True:
            session_id = np.random.choice(list(self.sessions.keys()))
            event = self.sessions[session_id].sample().iloc[0]

            for _, row in self.data.iterrows():
                if row['session_id'] == session_id and row['event_id'] == event['event_id']:
                    break
            else:
                break

        return event


    def cache_event_labels(self):
        self.event_data = pd.DataFrame(self.event_data)
        self.data = pd.concat([self.data, self.event_data], ignore_index=True)


    def save_data(self):
        self.data['event_id'] = self.data['event_id'].astype(int)
        self.data['subevent_id'] = self.data['subevent_id'].astype(int)
        self.data.to_csv(self.label_data_path, sep='\t', index=False)


    def append(self, response):
        self.event_data['session_id'].append(self.event['session_id'])
        self.event_data['event_id'].append(self.event['event_id'])
        self.event_data['subevent_id'].append(self.curr_subevent_id)
        self.event_data['t_start'].append(self.curr_subevent_id * self.subevent_time)
        self.event_data['t_end'].append((self.curr_subevent_id + 1) * self.subevent_time)
        self.event_data['label'].append(labels[int(response)])
        self.event_data['date_labeled'].append(datetime.datetime.now())


    def increment_subevent(self):
        if self.curr_subevent_id == self.num_subevents - 1:
            return False

        self.curr_subevent_id += 1
        self.curr_subevent = self.subevents[self.curr_subevent_id]

        return True


    def set_subevents(self, event):
        self.subevents = []
        self.num_subevents = int(event.t_len // self.subevent_time)
        subevent_len = int(self.subevent_time * maple.RATE)

        event_audio = event['audio']
        for i in range(self.num_subevents):
            self.subevents.append(event_audio[i*subevent_len: (i+1)*subevent_len])

        self.curr_subevent = self.subevents[0]
        self.curr_subevent_id = 0


    def play_event(self, event):
        # Normalize volumes so barks aren't too loud, and grrrs aren't too soft
        audio = np.copy(event['audio']).astype(float)
        audio *= 10000 / np.max(audio)
        audio = audio.astype(maple.ARRAY_DTYPE)

        sd.play(audio, blocking=True)


    def play_subevent(self, subevent):
        # Normalize volumes so barks aren't too loud, and grrrs aren't too soft
        audio = np.copy(subevent).astype(float)
        audio *= 10000 / np.max(audio)
        audio = audio.astype(maple.ARRAY_DTYPE)

        sd.play(audio, blocking=True)


    def filter(self, max_t_len=10):
        for events in self.sessions.values():
            events = events[events['t_len'] <= max_t_len]


class Train(object):
    def __init__(self, args=argparse.Namespace()):
        """Train a model based off label data

        Parameters
        ==========
        args : argparse.Namespace
            An `argparse.Namespace` object with `label_data` and `model_dir` paths. `label_data` is
            a string pointing to the label data filepath to be used for training. `model_dir` is a string
            pointing to a directory where the model will be stored. The directory must not exist.
        """

        self.label_dict = {v: k for k, v in labels.items()}

        if args.label_data is None:
            raise Exception("Must provide --label-data in order to train!")

        A = lambda x: args.__dict__.get(x, None)
        self.trans = A('transformation')
        self.log = A('log_the_data')
        if self.log and self.trans not in ['spectrogram', 'fourier']:
            raise Exception("Cannot log transform data unless --transformation is in {'spectrogram', 'fourier'}")

        self.model_dir = A('model_dir')

        self.label_data_path = Path(args.label_data)
        self.label_data = pd.read_csv(self.label_data_path, sep='\t')

        self.subevent_time = self.infer_subevent_time()

        self.dbs = {}
        session_ids = self.label_data['session_id'].unique()
        for session_id in session_ids:
            self.dbs[session_id] = data.SessionAnalysis(name=session_id)

        self.norm = True
        self.model = None


    def run(self, disconnect_dbs=True):
        """Run the training procedure

        This method glues the procedure together.

        Parameters
        ==========
        disconnect_dbs : bool, True
            Disconnect from the session databases at the end of this method.
        """

        self.setup_dir()
        self.prep_data(transformation=self.trans)

        self.fit_data()

        self.save_model(self.model_dir / 'model.dat')

        if disconnect_dbs:
            self.disconnect_dbs()


    def setup_dir(self):
        self.model_dir = Path(self.model_dir)
        if self.model_dir.exists():
            raise Exception(f"Will not overwrite folder {self.model_dir}, since it already exists")
        else:
            self.model_dir.mkdir()


    def disconnect_dbs(self):
        """Disconnect the SQL connections for all sessions databases"""
        for db in self.dbs.values():
            db.disconnect()


    def infer_subevent_time(self):
        """Returns the subevent time (chunk size) used for the labeled data"""
        return (self.label_data['t_end'] - self.label_data['t_start']).value_counts().index[0]


    def get_event_audio(self, session_id, event_id):
        """Returns the audio data for an event

        Parameters
        ==========
        session_id : str
            The ID of the session that the event is in
        event_id : int
            The ID of the event


        Returns
        =======
        output : numpy array, dtype = np.int16
            The audio data.
        """

        return self.dbs[session_id].get_event_audio(event_id)


    def get_subevent_audio(self, session_id, event_id, subevent_id):
        """Returns the audio data for a subevent (chunk)

        Parameters
        ==========
        session_id : str
            The ID of the session that the event is in
        event_id : int
            The ID of the event
        subevent_id : int
            The ID of the subevent

        Returns
        =======
        output : numpy array, dtype = np.int16
            The audio data
        """

        event_audio = self.get_event_audio(session_id, event_id)

        subevent_len = int(self.subevent_time * maple.RATE)
        subevent_audio = event_audio[subevent_id * subevent_len: (subevent_id + 1) * subevent_len]

        return subevent_audio


    def get_subevent_spectrogram(self, session_id, event_id, subevent_id):
        """Calculates the spectrogram of a subevent

        Parameters
        ==========
        session_id : str
            The ID of the session that the event is in
        event_id : int
            The ID of the event
        subevent_id : int
            The ID of the subevent

        Returns
        =======
        output : numpy array
            A flattened spectrogram array.
        """

        subevent_audio = self.get_subevent_audio(session_id, event_id, subevent_id)
        return audio.get_spectrogram(subevent_audio, flatten=True)[2]


    def get_subevent_fourier(self, session_id, event_id, subevent_id):
        """Calculates the fourier amplitude spectrum of a subevent

        Parameters
        ==========
        session_id : str
            The ID of the session that the event is in
        event_id : int
            The ID of the event
        subevent_id : int
            The ID of the subevent

        Returns
        =======
        output : numpy array
            A fourier amplitude array.
        """

        subevent_audio = self.get_subevent_audio(session_id, event_id, subevent_id)
        return audio.get_fourier(subevent_audio)[0]


    def get_audio_length(self):
        """Returns the dimension of an audio chunk based on the subevent time and sampling rate"""

        data = self.label_data.iloc[0]
        return len(self.get_subevent_audio(
            session_id = data['session_id'],
            event_id = data['event_id'],
            subevent_id = data['subevent_id'],
        ))


    def get_spectrogram_length(self):
        """Returns the dimension of a spectrogram chunk based on the subevent time and sampling rate"""

        data = self.label_data.iloc[0]
        return len(self.get_subevent_spectrogram(
            session_id = data['session_id'],
            event_id = data['event_id'],
            subevent_id = data['subevent_id'],
        ))


    def get_fourier_length(self):
        """Returns the dimension of a fourier chunk based on the subevent time and sampling rate"""

        data = self.label_data.iloc[0]
        return len(self.get_subevent_fourier(
            session_id = data['session_id'],
            event_id = data['event_id'],
            subevent_id = data['subevent_id'],
        ))


    def prep_data(self, transformation='spectrogram'):
        """Establishes the training and validation datasets

        This method sets the attributes `self.X`, and `self.y`

        Parameters
        ==========
        transformation : str, 'spectrogram'
            Pick any of {'spectrogram', 'none', 'fourier'}.
        """

        a = self.label_data.shape[0]

        if transformation == 'spectrogram':
            transformation_fn = self.get_subevent_spectrogram
            b = self.get_spectrogram_length()
        elif transformation == 'none':
            transformation_fn = self.get_subevent_audio
            b = self.get_audio_length()
        elif transformation == 'fourier':
            transformation_fn = self.get_subevent_fourier
            b = self.get_fourier_length()
        else:
            raise Exception(f"transformation '{transformation}' not implemented.")

        self.X = np.zeros((a, b))
        self.y = np.zeros(a).astype(int)

        for i, subevent in self.label_data.iterrows():
            label = self.label_dict[subevent['label']]
            self.y[i] = label

            self.X[i, :] = transformation_fn(
                session_id = subevent['session_id'],
                event_id = subevent['event_id'],
                subevent_id = subevent['subevent_id'],
            )

        if self.log:
            self.X = np.log2(self.X)

        if self.norm:
            self.X = (self.X - self.X.mean(axis=1)[:, None]) / self.X.std(axis=1)[:,None]


    def split_data(self, train_frac=0.8):
        """This method sets the attributes `self.X_train`, `self.y_train`, `self.X_validate`, `self.y_validate`"""

        a = len(self.y)

        indices = np.arange(a)

        # Shuffles in place
        np.random.shuffle(indices)

        shuffled_indices = indices
        del indices

        self.X_train = self.X[shuffled_indices[:int(a*train_frac)], :]
        self.y_train = self.y[shuffled_indices[:int(a*train_frac)]]

        self.X_validate = self.X[shuffled_indices[int(a*train_frac):], :]
        self.y_validate = self.y[shuffled_indices[int(a*train_frac):]]


    def fit_data(self, param_grid=None, cv=20):
        """Trains a random forest classifier and calculates a model score.

        This method trains a bunch of models over a small subset of hyperparameter space based on an
        ad-hoc analysis described here:
        ekiefl.github.io/2021/03/14/maple-classifier/#-hyperparameter-tuning

        For each model setting, a 5-fold cross validation is used. When the most accurate model is
        determined, it is stored as `self.model` and the 5-fold cross validation accuracy is stored
        as self.model.xval_score_.
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [200],
                'criterion': ['gini', 'entropy'],
            },

        model_search = GridSearchCV(
            estimator = RandomForestClassifier(oob_score=True),
            param_grid = param_grid,
            cv = cv,
            verbose = 2,
            n_jobs = -1,
        )
        model_search.fit(self.X, self.y)

        self.model = model_search.best_estimator_
        self.model.xval_score_ = model_search.best_score_
        print(f'Model CV score: {self.model.xval_score_}')


    def save_model(self, filepath):
        """Saves `self.model` as a file with using `joblib.dump`

        Saves `self.model`, which is a `sklearn.ensemble.RandomForestClassifier` object, to
        `filepath`.  Before saving, some the `sample_rate`, `subevent_time`, `subevent_len`, and
        whether the spectrogram was log-transformed (`log`) are stored as additional attributes of
        `self.model`.

        Parameters
        ==========
        filepath : str, Path-like
            Stores the model with `joblib.dump`
        """

        self.model.log = self.log
        self.model.norm = self.norm
        self.model.trans = self.trans
        self.model.subevent_time = self.subevent_time
        self.model.sample_rate = maple.RATE
        self.model.subevent_len = int(self.model.subevent_time * self.model.sample_rate)
        joblib.dump(self.model, filepath)


class Classifier(object):
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise Exception(f'{path} does not exist')

        self.model = joblib.load(path)


    def predict(self, event_audio, as_label=False):
        """Given an arbitrary audio length, predict the class"""

        num_chunks = int(len(event_audio) / self.model.subevent_len)
        if not num_chunks:
            return 'none' if as_label else 0

        data = np.zeros((num_chunks, self.model.n_features_))
        for i in range(num_chunks):
            audio_chunk = event_audio[i * self.model.subevent_len: (i + 1) * self.model.subevent_len]
            data[i, :] = self.transform(audio_chunk)

        chunk_predictions = self.model.predict(data)

        # most common
        prediction = np.bincount(chunk_predictions).argmax()
        return labels[prediction] if as_label else prediction


    def transform(self, audio_chunk):
        if self.model.trans == 'spectrogram':
            data = audio.get_spectrogram(audio_chunk, fs=self.model.sample_rate, flatten=True)[2]
        elif self.model.trans == 'fourier':
            data = audio.get_fourier(audio_chunk, fs=self.model.sample_rate)[0]
        else:
            data = np.copy(audio_chunk)

        if self.model.log:
            data = np.log2(data)

        if self.model.norm:
            data = (data - data.mean()) / data.std()

        if np.isnan(data).all():
            # In rare cases an audio chunk may have all zeros, which result in nan's that break
            # the predict method.
            data = np.zeros(len(data))

        return data


class Classify(Classifier):
    """Update the 'class' column of the 'events' table in a list of sessions"""

    def __init__(self, args):
        if args.model_dir is None:
            raise Exception("You must provide a --model-dir")

        if args.session_paths is None:
            self.session_paths = sorted(maple.db_dir.glob('*/events.db'))
        else:
            self.session_paths = sorted([Path(x.strip()) for x in open(args.session_paths).readlines()])
        self.sessions = {}

        path = Path(args.model_dir) / 'model.dat'
        Classifier.__init__(self, path)


    def load_session_dbs(self):
        self.sessions = {}
        for session_path in self.session_paths:
            print(f'Loading session path {session_path}')
            session_db = data.SessionAnalysis(path=session_path)

            if session_db.dog.empty:
                continue

            session_id = session_db.meta[session_db.meta['key'] == 'id'].iloc[0]['value']
            self.sessions[session_id] = session_db


    def run(self):
        self.load_session_dbs()

        for session_id, db in self.sessions.items():
            classes = []
            for event_id in db.dog['event_id']:
                event_class = self.predict(db.get_event_audio(event_id), as_label=True)
                classes.append(event_class)

            db.dog['class'] = classes
            db.insert_rows_from_dataframe('events', db.dog, replace=True)

        self.disconnect_dbs()


    def disconnect_dbs(self):
        for db in self.sessions.values():
            db.disconnect()


