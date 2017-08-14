import h5py
import numpy as np
import time

fileName = '/gpfs/cfel/fsds/labs/processed/grid/m2_dark_00000.nxs'
dataPathInFile = '/entry/instrument/detector/data'
cellToExport = 175

saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/m2_spatial_position3_dark.h5'

f = h5py.File(fileName, 'r', libver='latest')

dataCount = int(f[dataPathInFile].shape[0] / 352/ 2)
saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 128, 512), compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 128, 512), compression=None, dtype='int16')

t = time.time()
print('start loading', fileName)
rawData = f[dataPathInFile][...]
print('took time:  ' + str(time.time() - t))

rawData.shape = (dataCount, 352, 2, 128, 512)

analog_tmp = rawData[:, cellToExport, 0, :, :]

t = time.time()
print('start saving analog to ', saveFileName)
dset_analog[...] = analog_tmp
print('took time:  ' + str(time.time() - t))

t = time.time()
print('flushing')
saveFile.flush()
print('took time:  ' + str(time.time() - t))

f.close()
