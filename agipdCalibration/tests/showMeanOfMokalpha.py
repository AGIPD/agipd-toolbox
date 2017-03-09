import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg



dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/mokalphaData_m3.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')
analog = dataFile['analog'][...]

# meanImage = np.mean(analog, axis = 0)
medianImage = np.median(analog, axis = 0)

pg.image(medianImage.transpose(1,0))

i = 1