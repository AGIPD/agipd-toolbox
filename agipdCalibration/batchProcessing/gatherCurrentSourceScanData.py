import h5py
import time
import sys
import numpy as np

# dataFileName = '/gpfs/cfel/fsds/labs/processed/m1_m233_drscsvr160_i80_00002.nxs'
# dataPathInFile = '/entry/instrument/detector/data'
#
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/currentSource_chunked.h5'

# dataFileName_column1and5 = sys.argv[1]
# dataFileName_column2and6 = sys.argv[2]
# dataFileName_column3and7 = sys.argv[3]
# dataFileName_column4and8 = sys.argv[4]
# dataPathInFile = '/entry/instrument/detector/data'
# saveFileName = sys.argv[5]

dataFileName_column1and5 = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col1and5_00000.nxs'
dataFileName_column2and6 = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col2and6_00000.nxs'
dataFileName_column3and7 = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col3and7_00000.nxs'
dataFileName_column4and8 = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/m1_cdslow_col4and8_00000.nxs'
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/drscs_step/combined.h5'

print('\n\n\nstart gatherCurrentSourceScanData')
print('saveFileName = ', saveFileName)
print('')

fileNames = (dataFileName_column1and5, dataFileName_column2and6, dataFileName_column3and7, dataFileName_column4and8)

f = h5py.File(fileNames[0], 'r', libver='latest')
dataCount = int(f[dataPathInFile].shape[0] / 2 / 352)
f.close()

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                      compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                       compression=None, dtype='int16')

analog = np.zeros((dataCount, 352, 128, 512), dtype='int16')
digital = np.zeros((dataCount, 352, 128, 512), dtype='int16')

for i in np.arange(4):
    t = time.time()
    print('start loading', fileNames[i])
    f = h5py.File(fileNames[i], 'r', libver='latest')
    rawData = f[dataPathInFile][..., 0:128, 0:512]
    print('took time:  ' + str(time.time() - t))

    t = time.time()
    print('start reshaping')
    rawData.shape = (dataCount, 352, 2, 128, 512)
    tmp = rawData[:, :, 0, :, :]
    analog[..., 0:64, np.arange(3-i, 512, 4)] = tmp[..., 0:64, np.arange(3-i, 512, 4)]
    analog[..., 64:, np.arange(i, 512, 4)] = tmp[..., 64:, np.arange(i, 512, 4)]
    tmp = rawData[:, :, 1, :, :]
    digital[..., 0:64, np.arange(3 - i, 512, 4)] = tmp[..., 0:64, np.arange(3 - i, 512, 4)]
    digital[..., 64:, np.arange(i, 512, 4)] = tmp[..., 64:, np.arange(i, 512, 4)]
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
