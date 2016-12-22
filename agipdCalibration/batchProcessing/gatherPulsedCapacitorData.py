import h5py
import numpy as np
import time
import glob
import sys

# folderName = '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/'
# loadModule = '3'
# saveModule = '4'
# saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/pulsedCapacitor_m' + saveModule + '_chunked.h5'

folderName = sys.argv[1]
loadModule = sys.argv[2]
saveFileName = sys.argv[3]
dataPathInFile = '/entry/instrument/detector/data'

print('\n\n\nstart gatherPulsedCapacitorData')
print('folderName = ', folderName)
print('loadModule = ', loadModule)
print('saveFileName = ', saveFileName)
print(' ')

totalTime = time.time()

fileName = folderName + 'row1/m' + loadModule + '_drspc_00000.nxs'
f = h5py.File(fileName, 'r', libver='latest')
dataCount = int(f[dataPathInFile].shape[0] / 2 / 352)
f.close()

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                      compression=None, dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 352, 128, 512), chunks=(dataCount, 352, 64, 64),
                                       compression=None, dtype='int16')

for row in np.arange(8):
    fileNamePattern = folderName + 'row' + str(row + 1) + '/m' + loadModule + '*'
    fileName = glob.glob(fileNamePattern)[0]

    # workaround for corrupt file
    if fileName == '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/drspc/row7/m3_drspc_00006.nxs':
        print('workaround!!!!!!! Skipping file ', fileName)
        continue

    columnsToLoadPerIteration = 512

    f = h5py.File(fileName, 'r', libver='latest')

    for column in np.arange(512 / columnsToLoadPerIteration).astype(int):
        x_slice = slice(int(column * columnsToLoadPerIteration), int((column + 1) * columnsToLoadPerIteration))
        y_slice = slice(0, 128)  # full data range

        t = time.time()
        print('start loading columns', x_slice.start, '-', x_slice.stop, 'in rows', y_slice.start, '-', y_slice.stop, 'from', fileName)
        rawData = f[dataPathInFile][..., y_slice, x_slice]
        print('took time:  ' + str(time.time() - t))

        rawData.shape = (dataCount, 352, 2, 128, columnsToLoadPerIteration)

        analog_tmp = rawData[:, :, 0, :, :]
        digital_tmp = rawData[:, :, 1, :, :]

        t = time.time()
        print('start saving analog columns', x_slice.start, '-', x_slice.stop, 'in rows', y_slice.start, '-', y_slice.stop, 'at', saveFileName)
        dset_analog[..., row:64:8, x_slice] = analog_tmp[..., (8 - (row + 1)):64:8, :]
        dset_analog[..., (64 + row)::8, x_slice] = analog_tmp[..., (64 + row)::8, :]
        print('took time:  ' + str(time.time() - t))
        t = time.time()
        print('start saving digital columns', x_slice.start, '-', x_slice.stop, 'in rows', y_slice.start, '-', y_slice.stop, 'at', saveFileName)
        dset_digital[..., row:64:8, x_slice] = digital_tmp[..., (8 - (row + 1)):64:8, :]
        dset_digital[..., (64 + row)::8, x_slice] = digital_tmp[..., (64 + row)::8, :]
        print('took time:  ' + str(time.time() - t))
        t = time.time()
        print('flushing')
        saveFile.flush()
        print('took time:  ' + str(time.time() - t))

    f.close()

print('gatherPulsedCapacitorData took time:  ', time.time() - totalTime, '\n\n')
