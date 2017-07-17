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
cgDir = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/clamped_gain/'
darkDir = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/dark/'
outDir = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/jenny_stash/GainBitCorrection/'

# Input file names
# For dark: tango splits into parts! this is the base of the name
# the files end with "part0000<n>.nxs" where <n> goes from 0 to nParts
nParts = 10
fileName_dark = 'M305_m8_dark_tint150ns_00000_part000' 
fileName_med = 'M305_m8_cg_medium_00000.nxs'
fileName_low = 'M305_m8_cg_low_00000.nxs'

# Complete path to input files
fDark = os.path.join(darkDir, fileName_dark)
fMed = os.path.join(cgDir, fileName_med)
fLow = os.path.join(cgDir, fileName_low)

dataPathInFile = '/entry/instrument/detector/data'

# Output file containing means and stdev for each gain stage
saveFileName = 'M314_Tm15_clampedGainData_testParts.h5'
saveFilePath = os.path.join(outDir, saveFileName)


# High gain from dark data
# have to calculate how many events first!
f = h5py.File(fDark + '00.nxs', 'r', libver='latest')
dataCountPerFile = int(f[dataPathInFile].shape[0] / 2 / 352) #how many per file
dataCount = dataCountPerFile * nParts # times nParts files for total
f.close()

# Loop over all nParts files, read in data
for j in np.arange(nParts):
    if j <= 9:
        fDark_j = fDark + '0' + str(j) + '.nxs'
    else:
        fDark_j = fDark + str(j) + '.nxs'
    print('start loading', dataPathInFile, 'from', fDark_j)
    f = h5py.File(fDark_j, 'r')
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
