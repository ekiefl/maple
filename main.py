#! /usr/bin/env python

import maple.data as data
import maple.routines as routines

import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('run', 'record', 'analyze'), help="What mode do you want to run in?")

    RUN = ap.add_argument_group('run', 'arguments for `run` mode')
    RUN.add_argument('-n', '--no-respond', action='store_true')

    ANALYZE = ap.add_argument_group('analyze', 'arguments for `analyze` mode')
    # FIXME help description lies
    ANALYZE.add_argument(
        '-s',
        '--session',
        default=None,
        help='Name of session to analyze. If not set, analysis will be ran for all sessions, as well as a meta-analysis'
    )

    RECORD = ap.add_argument_group('record', 'arguments for `record` mode')

    args = ap.parse_args()

    if args.mode == 'run':
        routines.MonitorDog(args).run()
    elif args.mode == 'analyze':
        data.DBAnalysis(args.session)
    elif args.mode == 'record':
        routines.RecordOwnerVoice(args).run()
