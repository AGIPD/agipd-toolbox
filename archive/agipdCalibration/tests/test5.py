import h5py
import sys
import numpy as np
import time

import matplotlib.pyplot as plt
import pyqtgraph as pg

fileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/clampedDigitalMeans.h5'


f = h5py.File(fileName, 'r')
clampedDigitalMeans = f['clampedDigitalMeans'][()]
clampedDigitalStandardDeviations = f['clampedDigitalStandardDeviations'][()]
f.close()




i = 1
