import h5py
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/calibration/current/cooled/drscs_150_no_scaling/m7_drscs_lc_no_scaling_15_00004_part00000.nxs'
dataPathInFile = '/entry/instrument/detector/data'

f = h5py.File(dataFileName, 'r', libver='latest')
rawData = f[dataPathInFile][..., 0:128, 0:512]

rawData.shape = (-1, 352, 2, 128, 512)

analog = rawData[:, :, 0, :, :]
digital = rawData[:, :, 1, :, :]

i = 0