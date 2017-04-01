import h5py
import time
import numpy as np

dataFileName = '/gpfs/cfel/fsds/labs/processed/m1_m233_drscsvr160_i80_00002.nxs'
dataPathInFile = '/entry/instrument/detector/data'

saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/currentSource_chunked.h5'

f = h5py.File(dataFileName, 'r', libver='latest')
dataCount = int(f[dataPathInFile].shape[0] / 2 / 352)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                      compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                       compression=None, dtype='int16')

t = time.time()
print('start loading')
rawData = f[dataPathInFile][..., 0:128, 0:512]
print('took time:  ' + str(time.time() - t))

rawData.shape = (dataCount, 352, 2, 128, 512)

t = time.time()
print('start saving')
dset_analog[...] = rawData[:, :, 0, :, :]
dset_digital[...] = rawData[:, :, 1, :, :]
saveFile.flush()
print('took time:  ' + str(time.time() - t))

f.close()
saveFile.close()