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


dataFileName = '/gpfs/cfel/fsds/labs/processed/calibration/processed/M303/temperature_40C/drscs/itestc80/M303_m6_drscs_itestc80_chunked.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')

consideredPixelsY = (0, 64)
consideredPixelsX = (0, 64)

print('loading data, rows ', consideredPixelsY[0], '-', consideredPixelsY[1], ' columns ', + consideredPixelsX[0], '-', consideredPixelsX[1],
      'from', dataFileName)
t = time.time()
analog = dataFile['/analog'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
digital = dataFile['/digital'][:, :, consideredPixelsY[0]:consideredPixelsY[1], consideredPixelsX[0]:consideredPixelsX[1]]
print('took time:  ', time.time() - t)
dataFile.close()

x = 1
y = 1
analog_local = analog[:, 3, y, x]
digital_local = digital[:, 3, y, x]

fitSlopesResult = fit3DynamicScanSlopes(analog_local, digital_local)

i = 0
