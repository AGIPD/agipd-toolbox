import h5py
import numpy as np
import sys
import time

# fileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/m1_M314_XRAY_FLUOR_00000.nxs'
# dataPathInFile = '/entry/instrument/detector/data'
#
# saveFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/mokalphaData_m1_new_v2.h5'

fileName = sys.argv[1]
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = sys.argv[2]

print('\n\n\nstart gatherXRayTubeData')
print('fileName = ', fileName)
print('dataPathInFile = ', dataPathInFile)
print('saveFileName = ', saveFileName)
print(' ')

totalTime = time.time()

f = h5py.File(fileName, 'r')
dataCount = int(f[dataPathInFile].shape[0] / 2)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 128, 512), dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 128, 512), dtype='int16')

print('start loading')
rawData = np.array(f[dataPathInFile])
print('loading done')
f.close()

analog = rawData[::2, ...]
digital = rawData[1::2, ...]

print('start saving')
dset_analog[...] = analog
dset_digital[...] = digital
print('saving done')

saveFile.flush()
saveFile.close()

print('gatherXRayTubeData took time:  ', time.time() - totalTime, '\n\n')
