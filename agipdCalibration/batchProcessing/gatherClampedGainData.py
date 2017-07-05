"""
High gain bit: from dark data
Med gain bit: from clamped gain
Low gain bit: from clamped gain

Clamped gain data format:
_______________________

Image memcell 0           \

Image memcell 1              \

.................                        /        x100 times for statistics

Image memcell 351       /

------------------------------ / 

"""

import h5py
import sys
import numpy as np
import time
import os


# Directories
cgDir = '/gpfs/cfel/fsds/labs/calibration/current/raw/302-303-314-305/m15/clamped_gain/'
darkDir = '/gpfs/cfel/fsds/labs/calibration/current/7-modules/temperature_m25C/dark/'
outDir = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/jenny_stash/GainBitCorrection/'

# Input file names
fileName_dark = 'M314_m5_dark_tint150ns_00012_part00000.nxs' 
fileName_med = 'm7_20170703_medium_00000.nxs'
fileName_low = 'm7_20170703_low_00000.nxs'

# Complete path to input files
fDark = os.path.join(darkDir, fileName_dark)
fMed = os.path.join(cgDir, fileName_med)
fLow = os.path.join(cgDir, fileName_low)

dataPathInFile = '/entry/instrument/detector/data'

# Output file containing means and stdev for each gain stage
saveFileName = 'M314_Tm15_clampedGainData_test.h5'
saveFilePath = os.path.join(outDir, saveFileName)


# High gain from dark data
print('start loading', dataPathInFile, 'from', fDark)
f = h5py.File(fDark, 'r')
rawData = f[dataPathInFile][..., 0:128, 0:512]
f.close()
print('loading done')

print(rawData.shape) 
rawData.shape = (-1, 352, 2, 128, 512) # Dark data contains analog and digital
darkGain = rawData[:, :, 1, :, :] # only take digital for gain bit


# Med gain from clamped gain
print('start loading', dataPathInFile, 'from', fMed)
f = h5py.File(fMed, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

rawData.shape = (-1, 352, 128, 512)
mediumGain = rawData


# Low gain from clamped gain
print('start loading', dataPathInFile, 'from', fLow)
f = h5py.File(fLow, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

rawData.shape = (-1, 352, 128, 512)
lowGain = rawData


# Save results
print('start saving', saveFilePath)
saveFile = h5py.File(saveFilePath, "w", libver='latest')
dset_darkGainData = saveFile.create_dataset("darkGainData", shape=darkGain.shape, compression=None, dtype='int16')
dset_mediumGainData = saveFile.create_dataset("mediumGainData", shape=mediumGain.shape, compression=None, dtype='int16')
dset_lowGainData = saveFile.create_dataset("lowGainData", shape=lowGain.shape, compression=None, dtype='int16')

dset_darkGainData[...] = darkGain
dset_mediumGainData[...] = mediumGain
dset_lowGainData[...] = lowGain

print('saving done')
