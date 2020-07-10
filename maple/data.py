#! /usr/bin/env python

import maple

import pandas as pd
import sqlite3
import datetime

from pathlib import Path


class DataBase(object):
    def __init__(self, db_path=None, new_database=False):
        maple.db_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            self.db_path = maple.db_dir / (self.get_default_db_id() + '.db')
        else:
            self.db_path = Path(db_path)

        self.db_id = self.db_path.stem

        if new_database and self.db_path.exists():
            raise ValueError(f"db {self.db_path} already exists.")
        elif not new_database and not self.db_path.exists:
            raise ValueError(f"db {self.db_path} doesn't exist.")

        self.conn = sqlite3.connect(self.db_path)
        self.conn.text_factory = str
        self.cursor = self.conn.cursor()

        if new_database:
            self.create_self()
            self.create_table('events', maple.db_structure['events']['names'], maple.db_structure['events']['types'])
            self.create_table('owner_events', maple.db_structure['owner_events']['names'], maple.db_structure['owner_events']['types'])


    def get_default_db_id(self):
        dt = datetime.datetime.now()
        return '_'.join([str(x) for x in [dt.date().year, dt.date().month, dt.date().day, dt.time().hour, dt.time().minute, dt.time().second]])


    def get_table_as_dataframe(self, table_name, columns_of_interest=None):
        table_structure = self.get_table_structure(table_name)

        if not columns_of_interest:
            columns_of_interest = table_structure

        results_df = pd.read_sql('''SELECT * FROM "%s"''' % table_name, self.conn, columns=table_structure)

        if table_name == 'events':
            # These require special treatment
            results_df['t_start'] = pd.to_datetime(results_df['t_start'])
            results_df['t_end'] = pd.to_datetime(results_df['t_end'])
            results_df['audio'] = results_df['audio'].apply(utils.convert_blob_to_array)

        return results_df[columns_of_interest]


    def insert_rows_from_dataframe(self, table_name, df):
        df = df[self.get_table_structure(table_name)]

        df.to_sql(
            table_name,
            self.conn,
            if_exists='append',
            dtype=self.get_table_columns_and_types(table_name),
            index=False
        )


    def create_self(self):
        self._exec('''CREATE TABLE self (key text, value text)''')
        self._exec('''INSERT INTO self VALUES(?,?)''', ('id', self.db_id))
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
        self.conn.commit()
        self.conn.close()
