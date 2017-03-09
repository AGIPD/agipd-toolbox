import h5py
import sys
import numpy as np
import time

# fileName = '/gpfs/cfel/fsds/labs/calibration/current/1Mpix_calib/wing2/dark/m1_dark_00000.nxs'
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkOffset_m2.h5'
fileName = '/gpfs/cfel/fsds/labs/processed/M213_dark/data_klyuev/m1_dark1kclk100im_00003.nxs'
saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/darkOffset_agipd11.h5'
# fileName = sys.argv[1]
# saveFileName = sys.argv[2]
print('\n\n\nstart batchProcessDarkCal')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

dataPathInFile = '/entry/instrument/detector/data'

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(352, 128, 512), dtype='int16')
dset_darkStandardDeviation = saveFile.create_dataset("darkStandardDeviation", shape=(352, 128, 512), dtype='float')

totalTime = time.time()

print('start loading', dataPathInFile, 'from', fileName)
f = h5py.File(fileName, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')

analog = rawData[::2, ...]
analog.shape = (-1, 352, 128, 512)

print('start computing means and standard deviations')
means = np.mean(analog, axis=0)
standardDeviations = np.empty((352, 128, 512))
for cell in np.arange(352):
    standardDeviations[cell, ...] = np.std(analog[:, cell, :, :].astype('float'), axis=0)
print('done computing means and standard deviations')

#print('\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!! WATCH OUT!!!! WORKAROUND!!!!! flipping DETECTOR !!!!! !!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n')
#means = means[...,::-1] #np.rot90(means, 2)
#standardDeviations =  standardDeviations[...,::-1] #np.rot90(standardDeviations, 2)

print('start saving results at', saveFileName)
dset_darkOffset[...] = means
dset_darkStandardDeviation[...] = standardDeviations
saveFile.flush()
print('saving done')

saveFile.close()

print('batchProcessDarkCal took time:  ', time.time() - totalTime, '\n\n')
