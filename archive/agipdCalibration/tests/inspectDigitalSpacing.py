import h5py
import numpy as np
from agipdCalibration.tests.h5py_display import h5disp

import matplotlib.pyplot as plt
import pyqtgraph as pg

workspacePath = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/'
dataFileName_digitalMeans = workspacePath + 'digitalMeans_currentSource.h5'

dataFile_digitalMeans = h5py.File(dataFileName_digitalMeans, 'r', libver='latest')

digitalMeans = dataFile_digitalMeans["digitalMeans"][...] #  shape=(352, 3, 128, 512)
digitalThresholds = dataFile_digitalMeans["digitalThresholds"][...] #  shape=(2, 352, 128, 512)
digitalStdDeviations = dataFile_digitalMeans["digitalStdDeviations"][...] #  shape=(352, 3, 128, 512)
digitalSpacingsSafetyFactors = dataFile_digitalMeans["digitalSpacingsSafetyFactors"][...] #  shape=(352, 3, 128, 512)

plt.hist(digitalMeans[:,0,...].flatten(), 200)
plt.hist(digitalMeans[:,1,...].flatten(), 200)
plt.hist(digitalMeans[:,2,...].flatten(), 200)

plt.show()