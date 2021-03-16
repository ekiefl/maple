#! /usr/bin/env python

import maple
import maple.data as data
import maple.routines as routines

import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Open up `config` to change run parameters")

    ap.add_argument('mode', choices=('run', 'record', 'analyze', 'label', 'train'), help="What mode do you want to run in?")
    ap.add_argument('-t', '--temp', action='store_true', help='Use to search for and save DB sessions in data/temp')
    ap.add_argument('-q', '--quiet', action='store_true', help='Be quieter')
    ap.add_argument('-r', '--randomize', action='store_true', help='Randomize whether to praise and scold')

    RUN = ap.add_argument_group('run', 'arguments for `run` mode')
    RUN.add_argument('-c', '--sound-check', action='store_true', help='Test sound output volumes for owner response before starting. Happens before calibration.')

    ANALYZE = ap.add_argument_group('analyze', 'arguments for `analyze` mode')
    ANALYZE.add_argument('-s', '--session', required=None, help='Name of session to analyze')
    ANALYZE.add_argument('-p', '--path', required=None, help='fullpath of session to analyze')
    ANALYZE.add_argument('-l', '--list', action='store_true', help='List all sources and exit')

    RECORD = ap.add_argument_group('record', 'arguments for `record` mode')

    LABEL = ap.add_argument_group('label/train', 'arguments for `label/train` mode')
    LABEL.add_argument('-S', '--session-paths', default=None, help='Path to list of session paths to label data from')
    LABEL.add_argument('-T', '--label-data', required=None, help='Filepath where data is stored. Will be created if it doesn\'t exist')

    args = ap.parse_args()

    # update args Namespace with config values
    d = vars(args)
    for section in maple.config:
        d.update(maple.config[section])

    if args.mode == 'run':
        routines.MonitorDog(args).run()
    elif args.mode == 'analyze':
        routines.Analysis(args).run()
    elif args.mode == 'record':
        routines.RecordOwnerVoice(args).run()
    elif args.mode == 'label':
        routines.LabelAudio(args).run()
    elif args.mode == 'train':
        routines.Train(args).run()
