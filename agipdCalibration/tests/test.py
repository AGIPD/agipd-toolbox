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

dataFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/mokalphaData.h5'

f = h5py.File(dataFileName, 'r', libver='latest')
analog = f['analog'][...]

getOnePhotonAdcCountsXRayTubeData(analog[:,8,125], applyLowpass=True, localityRadius=801, lwopassSamplePointsCount=1000)

plt.plot(analog[:,8,125],'.')
plt.show()

i = 1

