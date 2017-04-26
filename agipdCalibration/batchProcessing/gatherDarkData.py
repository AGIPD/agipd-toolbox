import h5py
import sys
import numpy as np
import time

# fileName = '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/dark/m1_dark_00000.nxs'
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkData_m2.h5'

fileName = sys.argv[1]
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = sys.argv[2]

print('\n\n\nstart gatherDarkData')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

totalTime = time.time()

print('start loading', dataPathInFile, 'from', fileName)
f = h5py.File(fileName, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

analog = rawData[::2, ...]
analog.shape = (-1, 352, 128, 512)
digital = rawData[::2, ...]
digital.shape = (-1, 352, 128, 512)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=analog.shape, compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=digital.shape, compression=None, dtype='int16')

print('start saving to ', saveFileName)
dset_analog[...] = analog
dset_digital[...] = digital
saveFile.close()
print('saving done')

print('gatherDarkData took time:  ', time.time() - totalTime, '\n\n')











