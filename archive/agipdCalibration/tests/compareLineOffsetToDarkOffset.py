import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m3.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
offset = dataFile['/darkOffset'][...]   #shape = (352, 128, 512)

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/lysozymeData_m3_darkcal_inSitu.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')
offset_inSitu = dataFile['/darkOffset'][...]   #shape = (352, 128, 512)

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/analogGains_m3.h5'
dataFile = h5py.File(dataFileName, 'r', libver='latest')
offset_pulsedCapacitor = dataFile['/anlogLineOffsets'][0, ...]   #shape = (352, 128, 512)


offset2D = np.empty([128*512, 11, 32])
offset2D = offset[:, 0:128, 0:512]
offset2D = offset2D.transpose(1, 2, 0)
offset2D = offset2D.reshape([128, 512, 11, 32])

offset2D_inSitu = np.empty([128*512, 11, 32])
offset2D_inSitu = offset_inSitu[:, 0:128, 0:512]
offset2D_inSitu = offset2D_inSitu.transpose(1, 2, 0)
offset2D_inSitu = offset2D_inSitu.reshape([128, 512, 11, 32])

offset2D_pulsedCapacitor = np.empty([128*512, 11, 32])
offset2D_pulsedCapacitor = offset_pulsedCapacitor[:, 0:128, 0:512]
offset2D_pulsedCapacitor = offset2D_pulsedCapacitor.transpose(1, 2, 0)
offset2D_pulsedCapacitor = offset2D_pulsedCapacitor.reshape([128, 512, 11, 32])

overviewOffset = np.empty((128 * 11, 512 * 32))
overviewOffset_inSitu = np.empty((128 * 11, 512 * 32))
overviewOffset_pulsedCapacitor = np.empty((128 * 11, 512 * 32))
for y in np.arange(128):
    for x in np.arange(512):
        overviewOffset[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = offset2D[y, x, ...]
        overviewOffset_inSitu[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = offset2D_inSitu[y, x, ...]
        overviewOffset_pulsedCapacitor[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = offset2D_pulsedCapacitor[y, x, ...]

pg.image(overviewOffset.transpose(1,0))
pg.image(overviewOffset_inSitu.transpose(1,0))
pg.image(overviewOffset_pulsedCapacitor.transpose(1,0))
# pg.image((overviewOffset-overviewOffset_inSitu).transpose(1,0))
i = 1