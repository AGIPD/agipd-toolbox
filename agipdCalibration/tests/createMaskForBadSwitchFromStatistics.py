import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg


# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m3.h5'
dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/darkOffset_agipd11.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
standardDeviation = dataFile['/darkStandardDeviation'][...]
offset = dataFile['/darkOffset'][...]   #shape = (352, 128, 512)

# standardDeviation2D = np.empty([64*64, 11, 32])
# offset2D = np.empty([64*64, 11, 32])

standardDeviation2D = standardDeviation.transpose(1,2,0).reshape([128, 512, 11, 32])
offset2D = offset.transpose(1,2,0).reshape([128, 512, 11, 32])

# tmp = standardDeviation[:,0:64,0:64]  #shape = (352, 64, 64)
# tmp = tmp.transpose(1,2,0) #shape = (64, 64, 352)
# tmp = tmp.reshape([64*64, 11, 32])

tmp1 = standardDeviation2D[0:64,0:64,...].reshape([64*64, 11, 32])
tmp2  = offset2D[0:64,0:64,...].reshape([64*64, 11, 32])

a = pg.image(tmp1.transpose(0,2,1))
b = pg.image(tmp2.transpose(0,2,1))


i = 0
