import h5py
import numpy as np
import time
import glob
import sys
from agipdCalibration.algorithms.rangeScansFitting import *

import matplotlib.pyplot as plt
import pyqtgraph as pg

dataFileName = '/gpfs/cfel/fsds/labs/processed/m1_m233_drscsvr160_i80_00002.nxs'
dataPathInFile = '/entry/instrument/detector/data'

f = h5py.File(dataFileName, 'r', libver='latest')
dataCount = int(f[dataPathInFile].shape[0] / 2 / 352)

t = time.time()
print('start loading')
rawData = f[dataPathInFile][..., 0:128, 0:512]
print('took time:  ' + str(time.time() - t))

rawData.shape = (dataCount, 352, 2, 128, 512)

analog = rawData[:, :, 0, :, :]
digital = rawData[:, :, 1, :, :]

#(fitLineParameters, digitalMeanValues, analogFitStdDevs, (digitalStdDev_highGain, digitalStdDev_mediumGain)) = fit2DynamicScanSlopes(analog[:, 35, 50, 40], digital[:, 35, 50, 40])

