import h5py
import sys
import numpy as np
import time

fileName = '/gpfs/cfel/fsds/labs/calibration/current/flood_field_burst/m8_Mo_tube_burst_data_00000.nxs'
saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/gathered_m8_Mo_tube_burst_data_00000.nxs.h5'
dataPathInFile = '/entry/instrument/detector/data'

# fileName = sys.argv[1]
# dataPathInFile = '/entry/instrument/detector/data'
# saveFileName = sys.argv[2]

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
digital = rawData[1::2, ...]
digital.shape = (-1, 352, 128, 512)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=analog.shape, compression=None, chunks=(1, 352, 128, 512), dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=digital.shape, compression=None, chunks=(1, 352, 128, 512), dtype='int16')

print('start saving to ', saveFileName)
for i in np.arange(0, analog.shape[0]):
    dset_analog[i, ...] = analog[i, ...]
    dset_digital[i, ...] = digital[i, ...]
saveFile.close()
print('saving done')

print('gatherDarkData took time:  ', time.time() - totalTime, '\n\n')
