#! /usr/bin/env python

import maple.data as data
import maple.routines as routines

import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'record', 'analyze'), help="What mode do you want to run in?")
    ap.add_argument('-t', '--temp', action='store_true', help='Use to search for and save DB sessions in data/temp')
    ap.add_argument('-q', '--quiet', action='store_true', help='Be quieter')

    RUN = ap.add_argument_group('run', 'arguments for `run` mode')
    RUN.add_argument('-n', '--no-respond', action='store_true')

    ANALYZE = ap.add_argument_group('analyze', 'arguments for `analyze` mode')
    ANALYZE.add_argument('-s', '--session', required=None, help='Name of session to analyze')
    ANALYZE.add_argument('-p', '--path', required=None, help='fullpath of session to analyze')
    ANALYZE.add_argument('-l', '--list', action='store_true', help='List all sources and exit')

    RECORD = ap.add_argument_group('record', 'arguments for `record` mode')

    args = ap.parse_args()

    if args.mode == 'run':
        routines.MonitorDog(args).run()
    elif args.mode == 'analyze':
        routines.Analysis(args).run()
    elif args.mode == 'record':
        routines.RecordOwnerVoice(args).run()
