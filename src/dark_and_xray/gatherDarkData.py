import h5py
import sys
import numpy as np
import time

#fileName = '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/XFEL_testdata/m1_1_dark_before_00000.nxs'
#saveFileName = '/gpfs/cfel/fsds/labs/processed/calibration_1.1/jenny_stash/PhotonSpacing/darkData_m1_M314.h5'

nParts = int(sys.argv[1])
fileName = sys.argv[2]
dataPathInFile = '/entry/instrument/detector/data'
saveFileName = sys.argv[3]

print('\n\n\nstart gatherDarkData')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

totalTime = time.time()

# Data split into several files - "parts"
# first calculate how many total events
# assumption - same number of events in each file!
print('start loading', dataPathInFile, 'from', fileName)
f = h5py.File(fileName + '00.nxs', 'r')
dataCountPerFile = int(f[dataPathInFile].shape[0] / 2 / 352) #how many per file
dataCount = dataCountPerFile * nParts #times nParts files for total
f.close()

analog = np.zeros((dataCount, 352, 128, 512), dtype='int16')
digital = np.zeros((dataCount, 352, 128, 512), dtype='int16')

# Loop over all nParts files, read in data
for j in np.arange(nParts):
    if j <= 9:
        fDark_j = fileName + '0' + str(j) + '.nxs'
    else:
        fDark_j = fileName + str(j) + '.nxs'
    print('start loading ', dataPathInFile, ' from ', fDark_j)
    f = h5py.File(fDark_j, 'r')
    rawData = f[dataPathInFile][()]
    print('loading done')

    print('start reshaping')
    rawData.shape = (dataCountPerFile, 352, 2, 128, 512)
    analog[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, :, :] = rawData[:, :, 0, :, :]
    digital[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, :, :] = rawData[:, :, 1, :, :]
    print('finished reshaping')

    f.close()
    print('finished loading all data')

#analog = rawData[::2, ...]
#analog.shape = (-1, 352, 128, 512)
#digital = rawData[1::2, ...]
#digital.shape = (-1, 352, 128, 512)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=analog.shape, compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=digital.shape, compression=None, dtype='int16')

print('start saving to ', saveFileName)
dset_analog[...] = analog
dset_digital[...] = digital
saveFile.close()
print('saving done')

print('gatherDarkData took time:  ', time.time() - totalTime, '\n\n')











