#! /usr/bin/env python

import maple.sound as sound

with sound.LiveStream() as s:
    s.start()
