import h5py
import numpy as np
import time
import glob
import sys
from agipdCalibration.algorithms.rangeScansFitting import *
from agipdCalibration.algorithms.xRayTubeDataFitting import *

import matplotlib.pyplot as plt
import pyqtgraph as pg

# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing80.h5'
#
# f = h5py.File(dataFileName, 'r', libver='latest')
# photonSpacing80 = f['photonSpacing'][...]
#
# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/photonSpacing200.h5'
#
# f = h5py.File(dataFileName, 'r', libver='latest')
# photonSpacing200 = f['photonSpacing'][...]
# pg.image(photonSpacing80.transpose())
# pg.image(photonSpacing200.transpose())
# pg.image((photonSpacing80-photonSpacing200).transpose())

# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_full/photonSpacing_m1.h5'
dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/photonSpacing_m6_xray_Mo_00000.h5'
f = h5py.File(dataFileName, 'r', libver='latest')
photonSpacing175 = f['photonSpacing'][...]

pg.image(photonSpacing175.transpose())
plt.imshow(photonSpacing175, vmin=30, vmax=130)
plt.show()

tmp = photonSpacing175.flatten()
failed = (sum(tmp > 150) + sum(tmp < 120))/tmp.size


# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/xRay80.h5'
# # dataFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col1and5_00000.nxs'
# # dataPathInFile = '/entry/instrument/detector/data'
#
# f = h5py.File(dataFileName, 'r', libver='latest')
# analog80 = f['analog'][...]
#
# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/xRay200.h5'
# f = h5py.File(dataFileName, 'r', libver='latest')
# analog200 = f['analog'][...]
#
# # rawData = f[dataPathInFile][..., 0:128, 0:512]
# # rawData.shape = (-1, 352, 2, 128, 512)
# # analog = rawData[:, :, 0, :, :]
#
#
# # plt.plot(analog[:, 8, 15],'.')
# # plt.show()



i = 1
