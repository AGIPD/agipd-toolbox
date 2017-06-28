import h5py
import sys
import numpy as np
import time

import matplotlib.pyplot as plt
import pyqtgraph as pg

# fileName_dark = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_30C/dark/M316_m6_dark_tint150ns_00000_part00002.nxs'
# fileName_medium = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_30C/clamped_gain/M316_m6_cg_medium_00000.nxs'
# fileName_low = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_30C/clamped_gain/M316_m6_cg_low_00001.nxs'
fileName_dark = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_m25C/dark/M316_m6_dark_tint150ns_00012_part00002.nxs'
fileName_medium = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_m25C/clamped_gain/M316_m6_cg_medium_00005.nxs'
fileName_low = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_m25C/clamped_gain/M316_m6_cg_low_00006.nxs'
dataPathInFile = '/entry/instrument/detector/data'

saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/clampedGainData.h5'

print('start loading', dataPathInFile, 'from', fileName_dark)
f = h5py.File(fileName_dark, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

rawData.shape = (-1, 352, 2, 128, 512)
darkGain = rawData[:, :, 1, :, :]

print('start loading', dataPathInFile, 'from', fileName_medium)
f = h5py.File(fileName_medium, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

rawData.shape = (-1, 352, 128, 512)
mediumGain = rawData

print('start loading', dataPathInFile, 'from', fileName_low)
f = h5py.File(fileName_low, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

rawData.shape = (-1, 352, 128, 512)
lowGain = rawData

print('start saving', saveFileName)
saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_darkGainData = saveFile.create_dataset("darkGainData", shape=darkGain.shape, compression=None, dtype='int16')
dset_mediumGainData = saveFile.create_dataset("mediumGainData", shape=mediumGain.shape, compression=None, dtype='int16')
dset_lowGainData = saveFile.create_dataset("lowGainData", shape=lowGain.shape, compression=None, dtype='int16')

dset_darkGainData[...] = darkGain
dset_mediumGainData[...] = mediumGain
dset_lowGainData[...] = lowGain

print('saving done')

