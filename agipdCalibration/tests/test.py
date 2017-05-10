import h5py
import numpy as np
import time
import glob
import sys
from agipdCalibration.algorithms.rangeScansFitting import *
from agipdCalibration.algorithms.xRayTubeDataFitting import *

import matplotlib.pyplot as plt
import pyqtgraph as pg

# dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/photonSpacing.h5'
#
# f = h5py.File(dataFileName, 'r', libver='latest')
# photonSpacing = f['photonSpacing'][...]
#
# pg.image(photonSpacing.transpose())

dataFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/combined.h5'
# dataFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col1and5_00000.nxs'
# dataPathInFile = '/entry/instrument/detector/data'

f = h5py.File(dataFileName, 'r', libver='latest')
analog = f['analog'][...]
# rawData = f[dataPathInFile][..., 0:128, 0:512]
# rawData.shape = (-1, 352, 2, 128, 512)
# analog = rawData[:, :, 0, :, :]


plt.plot(analog[:, 10, 8, 0],'.')
plt.show()

i = 1

