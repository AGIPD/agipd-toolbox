import h5py
import sys
import numpy as np
import time

import matplotlib.pyplot as plt
import pyqtgraph as pg

filePath = '/gpfs/cfel/fsds/labs/processed/calibration/processed/M305/temperature_m15C/clamped_gain/'
fileName = filePath + 'M305_m8_clampedGainData.h5'
saveFileName = filePath + 'M305_m8_clampedDigitalMeans.h5'
# fileName = sys.argv[1]
# saveFileName = sys.argv[2]

print('\n\n\nstart batchProcessClampedGainData')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

totalTime = time.time()

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_clampedDigitalMeans = saveFile.create_dataset("clampedDigitalMeans", shape=(3, 352, 128, 512), compression=None, dtype='float32')
dset_clampedDigitalStandardDeviations = saveFile.create_dataset("clampedDigitalStandardDeviations", shape=(3, 352, 128, 512), compression=None, dtype='float32')

f = h5py.File(fileName, 'r')

print('start loading', '/darkGainData', 'from', saveFileName)
data = f['darkGainData'][()]
print('loading done')

print('start computing means and standard deviations')
meansHigh = np.mean(data, axis=0)
standardDeviationsHigh = np.empty((352, 128, 512))
for cell in np.arange(352):
    standardDeviationsHigh[cell, ...] = np.std(data[:, cell, :, :].astype('float'), axis=0)
print('done computing means and standard deviations')

dset_clampedDigitalMeans[0, ...] = meansHigh
dset_clampedDigitalStandardDeviations[0, ...] = standardDeviationsHigh

print('start loading', '/mediumGainData', 'from', saveFileName)
data = f['mediumGainData'][()]
print('loading done')

print('start computing means and standard deviations')
meansMedium = np.mean(data, axis=0)
standardDeviationsMedium = np.empty((352, 128, 512))
for cell in np.arange(352):
    standardDeviationsMedium[cell, ...] = np.std(data[:, cell, :, :].astype('float'), axis=0)
print('done computing means and standard deviations')

dset_clampedDigitalMeans[1, ...] = meansMedium
dset_clampedDigitalStandardDeviations[1, ...] = standardDeviationsMedium

print('start loading', '/lowGainData', 'from', saveFileName)
data = f['lowGainData'][()]
print('loading done')

print('start computing means and standard deviations')
meansLow = np.mean(data, axis=0)
standardDeviationsLow = np.empty((352, 128, 512))
for cell in np.arange(352):
    standardDeviationsLow[cell, ...] = np.std(data[:, cell, :, :].astype('float'), axis=0)
print('done computing means and standard deviations')
#print('start workaround for low means and standard deviations')
#meansLow = meansHigh + 2*(meansMedium - meansHigh)
#standardDeviationsLow = standardDeviationsMedium
#print('done workaround for low means and standard deviations')

dset_clampedDigitalMeans[2, ...] = meansLow
dset_clampedDigitalStandardDeviations[2, ...] = standardDeviationsLow

print('batchProcessDarkData took time:  ', time.time() - totalTime, '\n\n')
