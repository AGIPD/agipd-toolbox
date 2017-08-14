import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg


dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing_m3.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')
photonSpacing = dataFile['photonSpacing'][...]
quality = dataFile['quality'][...]

# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing_m4_inSitu.h5'
# dataFile = h5py.File(dataFileName, 'r', libver='latest')
# photonSpacing_inSitu = dataFile['photonSpacing'][...]

a = pg.image(photonSpacing.transpose(1,0))
# b = pg.image(photonSpacing_inSitu.transpose(1,0))

c = pg.image(quality.transpose(1,0))

i = 1




