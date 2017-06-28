import h5py
import time
import sys
import numpy as np

# dataFileNameRoot_column1and5 = '/gpfs/cfel/fsds/labs/calibration/current/m1_drscs_with_scaling_RT_15_00000_part0000'
# dataFileNameRoot_column2and6 = '/gpfs/cfel/fsds/labs/calibration/current/m1_drscs_with_scaling_RT_26_00001_part0000'
# dataFileNameRoot_column3and7 = '/gpfs/cfel/fsds/labs/calibration/current/m1_drscs_with_scaling_RT_37_00002_part0000'
# dataFileNameRoot_column4and8 = '/gpfs/cfel/fsds/labs/calibration/current/m1_drscs_with_scaling_RT_48_00003_part0000'
# clampedGainMediumFileName = sys.argv[4]
# clampedGainLowFileName = sys.argv[5]
# darkDataFileName = sys.argv[6]
# dataPathInFile = '/entry/instrument/detector/data'
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/currentSource_chunked.h5'

dataFileNameRoot_column1and5 = sys.argv[1]
dataFileNameRoot_column2and6 = sys.argv[2]
dataFileNameRoot_column3and7 = sys.argv[3]
dataFileNameRoot_column4and8 = sys.argv[4]
clampedGainMediumFileName = sys.argv[4]
clampedGainLowFileName = sys.argv[5]
darkDataFileName = sys.argv[6]
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = sys.argv[7]

print('\n\n\nstart gatherCurrentSourceScanData')
print('saveFileName = ', saveFileName)
print('')

fileNamesRoots = (dataFileNameRoot_column1and5, dataFileNameRoot_column2and6, dataFileNameRoot_column3and7, dataFileNameRoot_column4and8)

f = h5py.File(fileNamesRoots[0] + '0.nxs', 'r', libver='latest')
dataCountPerFile = int(f[dataPathInFile].shape[0] / 2 / 352)
dataCount = dataCountPerFile * 10
f.close()

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                      compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                       compression=None, dtype='int16')

analog = np.zeros((dataCount, 352, 128, 512), dtype='int16')
digital = np.zeros((dataCount, 352, 128, 512), dtype='int16')

for i in np.arange(4):
    for j in np.arange(10):
        t = time.time()
        fileName = fileNamesRoots[i] + str(j) + '.nxs'
        print('start loading', fileName)
        f = h5py.File(fileName, 'r', libver='latest')
        rawData = f[dataPathInFile][..., 0:128, 0:512]
        print('took time:  ' + str(time.time() - t))

        t = time.time()
        print('start reshaping')
        rawData.shape = (dataCountPerFile, 352, 2, 128, 512)
        tmp = rawData[:, :, 0, :, :]
        analog[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, 0:64, np.arange(3 - i, 512, 4)] = tmp[..., 0:64, np.arange(3 - i, 512, 4)]
        analog[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, 64:, np.arange(i, 512, 4)] = tmp[..., 64:, np.arange(i, 512, 4)]
        tmp = rawData[:, :, 1, :, :]
        digital[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, 0:64, np.arange(3 - i, 512, 4)] = tmp[..., 0:64, np.arange(3 - i, 512, 4)]
        digital[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, 64:, np.arange(i, 512, 4)] = tmp[..., 64:, np.arange(i, 512, 4)]
        print('took time:  ' + str(time.time() - t))

        f.close()

t = time.time()
print('')
print('start saving')
dset_analog[...] = analog
dset_digital[...] = digital
saveFile.flush()
print('took time:  ' + str(time.time() - t))
saveFile.close()
