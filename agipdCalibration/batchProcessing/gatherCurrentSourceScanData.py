import h5py
import time
import sys
import numpy as np

# dataFileNameRoot_column1and5 = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/drscs/itestc20/M305_m8_drscs_itestc20_col15_00009_part000'
# dataFileNameRoot_column2and6 = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/drscs/itestc20/M305_m8_drscs_itestc20_col26_00010_part000'
# dataFileNameRoot_column3and7 = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/drscs/itestc20/M305_m8_drscs_itestc20_col37_00011_part000'
# dataFileNameRoot_column4and8 = '/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/drscs/itestc20/M305_m8_drscs_itestc20_col48_00012_part000'
# dataPathInFile = '/entry/instrument/detector/data'
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/gathered_M305_m8_drscs_itestc20_-15degree_chunked.h5'

dataFileNameRoot_column1and5 = sys.argv[1]
dataFileNameRoot_column2and6 = sys.argv[2]
dataFileNameRoot_column3and7 = sys.argv[3]
dataFileNameRoot_column4and8 = sys.argv[4]
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = sys.argv[5]

print('\n\n\nstart gatherCurrentSourceScanData')
print('saveFileName = ', saveFileName)
print('')

fileNamesRoots = (dataFileNameRoot_column1and5, dataFileNameRoot_column2and6, dataFileNameRoot_column3and7, dataFileNameRoot_column4and8)

f = h5py.File(fileNamesRoots[0] + '00.nxs', 'r', libver='latest')
dataCountPerFile = int(f[dataPathInFile].shape[0] / 2 / 352)
dataCount = dataCountPerFile * 13
f.close()

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                      compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                       compression=None, dtype='int16')

analog = np.zeros((dataCount, 352, 128, 512), dtype='int16')
digital = np.zeros((dataCount, 352, 128, 512), dtype='int16')

for i in np.arange(4):
    for j in np.arange(13):
        t = time.time()
        if j <= 9:
            fileName = fileNamesRoots[i] + '0' + str(j) + '.nxs'
        else:
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
