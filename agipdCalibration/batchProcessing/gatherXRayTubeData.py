import h5py
import numpy as np
import sys
import time

# fileName = '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/xray/m2_xray_00001.nxs'
# dataPathInFile = '/entry/instrument/detector/data'
#
# saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/mokalphaData_m1_00001.nxs'

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
