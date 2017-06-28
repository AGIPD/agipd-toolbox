import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg

workspacePath = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/'
dataFileName_analogGains = workspacePath + 'analogGains_currentSource_gainsFromCS.h5'
dataFileName_digitalMeans = workspacePath + 'digitalMeans_currentSource_gainsFromCS.h5'

dataFile_analogGains = h5py.File(dataFileName_analogGains, 'r', libver='latest')
dataFile_digitalMeans = h5py.File(dataFileName_digitalMeans, 'r', libver='latest')

analogGains = dataFile_analogGains["analogGains"][...]  # shape=(3, 352, 128, 512)
analogLineOffsets = dataFile_analogGains["anlogLineOffsets"][...]  # shape=(3, 352, 128, 512)
analogFitStdDevs = dataFile_analogGains["analogFitStdDevs"][...]  # shape=(3, 352, 128, 512)

digitalMeans = dataFile_digitalMeans["digitalMeans"][...]  # shape=(352, 3, 128, 512)
digitalThresholds = dataFile_digitalMeans["digitalThresholds"][...]  # shape=(2, 352, 128, 512)
digitalStdDeviations = dataFile_digitalMeans["digitalStdDeviations"][...]  # shape=(352, 3, 128, 512)
digitalSpacingsSafetyFactors = dataFile_digitalMeans["digitalSpacingsSafetyFactors"][...]  # shape=(352, 3, 128, 512)

dataFile_digitalMeans.close()
dataFile_analogGains.close()

#dataToPlot = [analogGains[0,...],analogGains[1,...],analogGains[2,...]]
dataToPlot = [digitalMeans[:, 0,...],digitalMeans[:, 1,...],digitalMeans[:, 2,...]]
# dataToPlot = [digitalStdDeviations[:, 0, ...], digitalStdDeviations[:, 1, ...], digitalStdDeviations[:, 2, ...]]
for i in np.arange(len(dataToPlot)):
    data = dataToPlot[i]
    data[data == float('inf')] = 0
    data = data.transpose(1, 2, 0)  # shape = (128, 512, 352)
    data = data.reshape([128, 512, 11, 32])

    overview = np.empty((128 * 11, 512 * 32))
    for y in np.arange(128):
        for x in np.arange(512):
            overview[y * 11:(y + 1) * 11, x * 32:(x + 1) * 32] = data[y, x, ...]

    pg.image(overview.transpose(1, 0))

i = 1
