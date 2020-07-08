#! /usr/bin/env python

import numpy as np


def calc_power(data):
    """Calculate the power of a discrete time signal"""

    return np.mean(np.abs(data))*2

