import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import time
import sys

from agipdCalibration.algorithms.rangeScansFitting import *

# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/analogGains_currentSource.h5'
# f = h5py.File(dataFileName, 'r', libver='latest')
# analogGains = f['analogGains'][...]
#
# analogGains_medium = analogGains[1, 170, :, :]
#
# pg.image(analogGains_medium.transpose())


dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/currentSource_chunked.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')

consideredPixelsY = (64, 128)
consideredPixelsX = (64, 128)

print('loading data, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
      'from', dataFileName)
t = time.time()
analog = dataFile['/analog'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
digital = dataFile['/digital'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
print('took time:  ', time.time() - t)
dataFile.close()

analog_local = analog[:, 170, 15, 3]
digital_local = digital[:, 170, 15, 3]

fitSlopesResult = fit3DynamicScanSlopes(analog_local, digital_local)

i = 0
